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
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.db.models import Contact, User
    from src.llm.base import LLMProvider

from src.db.session import get_session
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
) -> InboxDecision:
    """Принимает решение по входящему сообщению.

    Порядок проверок (первое совпадение побеждает):
    1. Folder filter — IGNORE
    2. Urgency → NOTIFY_URGENT
    3. Draft suggestion → DRAFT_SUGGEST
    4. Smart digest → QUEUE_FOR_DIGEST
    5. По умолчанию → SILENT_LOG
    """
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

    # Построить провайдера, если не передан
    if provider is None and owner.settings.llm_provider:
        # Провайдер требует сессию — создаём временную

        async with get_session() as _session:
            provider = await build_provider(_session, owner)

    # 2. Классификация срочности (LLM если есть провайдер, иначе эвристика)
    from src.core.contacts.urgency_classifier import classify_urgency

    urgency = await classify_urgency(
        message_text,
        provider=provider,
        sender_name=sender_name,
    )

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
    monitored = json.loads(settings.monitored_folders)
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
