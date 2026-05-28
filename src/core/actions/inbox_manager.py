"""InboxManager: единая точка принятия решений по входящим сообщениям.

Определяет, что делать с каждым входящим сообщением:
- Срочное → уведомить владельца
- Авто-ответ → отправить авто-ответ (если включён и владелец оффлайн)
- Черновик → предложить черновик ответа
- В дайджест → отложить до следующего дайджеста
- Игнорировать → только сохранить в БД
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.db.models import Contact, User
    from src.llm.base import LLMProvider

from src.db.session import get_session
from src.llm.base import TaskType
from src.llm.router import build_provider

logger = logging.getLogger(__name__)


class InboxAction(str, Enum):
    """Действие, которое нужно выполнить с входящим сообщением."""

    AUTO_REPLY = "auto_reply"
    DRAFT_SUGGEST = "draft_suggest"
    NOTIFY_URGENT = "notify_urgent"
    QUEUE_FOR_DIGEST = "queue_for_digest"
    SILENT_LOG = "silent_log"
    IGNORE = "ignore"


@dataclass
class InboxDecision:
    """Решение, принятое InboxManager."""

    action: InboxAction
    reason: str = ""
    confidence: float = 1.0
    extra: dict = field(default_factory=dict)


async def process_incoming(
    message_text: str,
    sender_name: str,
    peer_id: int,
    owner: User,
    contact: Contact | None,
    provider: LLMProvider | None = None,
    *,
    is_private: bool = True,
) -> InboxDecision:
    """Принимает решение по входящему сообщению.

    Порядок проверок (первое совпадение побеждает):
    1. Folder filter — IGNORE
    2. Urgency → NOTIFY_URGENT
    3. Draft suggestion → DRAFT_SUGGEST
    4. Smart digest → QUEUE_FOR_DIGEST
    5. По умолчанию → SILENT_LOG
    """
    # 0. Group/channel messages are never urgent
    urgency_limit: str = "urgent"
    if not is_private:
        urgency_limit = "normal"

    # 1. Folder filter: если контакт не в отслеживаемых папках — игнорируем
    if _is_filtered_by_folder(owner, contact):
        return InboxDecision(
            action=InboxAction.IGNORE,
            reason="Контакт не входит в отслеживаемые папки",
            confidence=1.0,
        )

    if not message_text:
        return InboxDecision(
            action=InboxAction.SILENT_LOG,
            reason="Сообщение без текста",
            confidence=1.0,
        )

    # 1.5 Reputation gate — unknown/new contacts can't be URGENT
    reputation = _check_reputation(contact, sender_name, message_text)
    if reputation == "spam":
        return InboxDecision(
            action=InboxAction.IGNORE,
            reason="Spam detected — message patterns match spam",
            confidence=0.95,
        )
    elif reputation == "unknown":
        if _URGENCY_RANK[urgency_limit] > _URGENCY_RANK["normal"]:
            urgency_limit = "normal"
    elif reputation == "low":
        if _URGENCY_RANK[urgency_limit] > _URGENCY_RANK["important"]:
            urgency_limit = "important"
    # trusted — keep current urgency_limit

    # Построить провайдера, если не передан
    if provider is None and owner.settings.llm_provider:
        # Провайдер требует сессию — создаём временную

        async with get_session() as _session:
            provider = await build_provider(
                _session, owner, task_type=TaskType.CLASSIFY
            )

    # 2. Классификация срочности (LLM если есть провайдер, иначе эвристика)
    from src.core.contacts.urgency_classifier import classify_urgency

    urgency = await classify_urgency(
        message_text,
        provider=provider,
        sender_name=sender_name,
    )

    # Apply reputation/group cap on urgency
    if urgency == "urgent" and _URGENCY_RANK[urgency_limit] < _URGENCY_RANK["urgent"]:
        urgency = urgency_limit
    elif urgency == "important" and urgency_limit == "normal":
        urgency = "normal"

    # 3. Срочное уведомление
    if urgency == "urgent" and owner.settings.urgent_notify_enabled:
        return InboxDecision(
            action=InboxAction.NOTIFY_URGENT,
            reason=f"Срочное сообщение от {sender_name}",
            confidence=0.9,
            extra={"urgency": "urgent"},
        )

    # 4. Проверка на черновик (только если черновики включены)
    if _should_check_draft(owner, message_text, provider, urgency):
        return InboxDecision(
            action=InboxAction.DRAFT_SUGGEST,
            reason=f"Предложить черновик для ответа {sender_name}",
            confidence=0.8,
            extra={"urgency": urgency},
        )

    # 5. Если дайджест включён — в очередь
    if owner.settings.smart_digest_enabled:
        return InboxDecision(
            action=InboxAction.QUEUE_FOR_DIGEST,
            reason="Отложено до следующего дайджеста",
            confidence=1.0,
            extra={"urgency": urgency},
        )

    # 6. По умолчанию — только сохранить в БД
    return InboxDecision(
        action=InboxAction.SILENT_LOG,
        reason="Обычное сообщение, обработано фоново",
        confidence=1.0,
        extra={"urgency": urgency},
    )


def _is_filtered_by_folder(owner: User, contact: Contact | None) -> bool:
    """Проверяет, нужно ли игнорировать сообщение из-за folder filter."""
    settings = owner.settings
    if not settings.monitor_only_selected_folders or not settings.monitored_folders:
        return False
    try:
        monitored = json.loads(settings.monitored_folders)
    except json.JSONDecodeError:
        logger.warning("Invalid monitored_folders JSON: %r", settings.monitored_folders)
        monitored = []
    if not monitored:
        return False
    if contact is None:
        return True  # контакт не найден — не можем проверить папки
    contact_folders = [
        f.strip() for f in (contact.folder_names or "").split(",") if f.strip()
    ]
    return not any(f in monitored for f in contact_folders)


def _should_check_draft(
    owner: User, text: str, provider: LLMProvider | None, urgency: str
) -> bool:
    """Проверяет, стоит ли предлагать черновик на это сообщение."""
    settings = owner.settings
    if not settings.draft_suggestions_enabled:
        return False
    if settings.draft_only_important and urgency == "normal":
        return False
    return True


_URGENCY_RANK = {
    "normal": 1,
    "important": 2,
    "urgent": 3,
}


def _check_reputation(contact, sender_name: str, message_text: str) -> str:
    """Check sender reputation. Returns: 'trusted', 'low', 'unknown', 'spam'."""
    if not contact:
        # Unknown contact — check if message looks like spam
        if _looks_like_spam(message_text):
            return "spam"
        return "unknown"

    # Known contact — check signals
    spam = _looks_like_spam(message_text)
    if spam:
        # Even known contacts can send spam (compromised accounts)
        return "low"

    # Has conversation history?
    # (contact in DB means they've been synced — trusted)
    return "trusted"


_SPAM_PATTERNS = [
    # Price/spam patterns
    r"\$\d+\.?\d*",  # "$1.95"
    r"\d+\s*(?:доллар|dollar|usd|eur|₽|руб)",  # "100 рублей"
    # Admin/manager spam
    r"admin\s+deal",  # "Admin Deal"
    r"top\s+\w+\s+provider",  # "Top GPT Provider"
    # Emoji overload (>3 unique emoji in message)
    r"(?:[\U0001F300-\U0001F9FF]){4,}",
    # Suspension/banned words
    r"suspension\s+issue",
    r"warranty|warranty",
    r"restock(?:ed)?",
]


def _looks_like_spam(text: str) -> bool:
    """Heuristic spam detection — no LLM needed."""
    if not text:
        return False
    t = text.lower()
    # Check spam patterns
    for pattern in _SPAM_PATTERNS:
        if re.search(pattern, t, re.IGNORECASE):
            return True
    # Emoji density: >20% emoji in short messages
    if len(text) < 100:
        emoji_count = sum(1 for c in text if ord(c) > 0x1F000)
        if emoji_count > 0 and emoji_count / len(text) > 0.2:
            return True
    # ALL_CAPS + short message (<30 chars) = likely spam
    if len(text) < 30 and text == text.upper() and text.isalpha():
        return True
    # URL + short message + no personal context
    if ("http" in t or "www." in t) and len(text) < 200:
        if not any(w in t for w in ("привет", "здравствуй", "как дела", "слушай")):
            return True
    return False
