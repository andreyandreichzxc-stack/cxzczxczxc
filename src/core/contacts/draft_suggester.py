"""Auto-draft suggestion logic: rate limit, urgency filter, draft generation."""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.db.models import UserSettings
    from src.llm.base import LLMProvider
    from src.db.models import Contact
    from src.db.models import Message

logger = logging.getLogger(__name__)

# in-memory rate limit: {user_id: deque of datetime}
_draft_timestamps: dict[int, deque] = {}


def _check_rate_limit(user_id: int, max_per_hour: int) -> bool:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    if user_id not in _draft_timestamps:
        _draft_timestamps[user_id] = deque()
    q = _draft_timestamps[user_id]
    # удалить старые (>1 часа)
    while q and now - q[0] > timedelta(hours=1):
        q.popleft()
    if len(q) >= max_per_hour:
        return False
    q.append(now)
    return True


async def should_suggest(
    settings: "UserSettings", user_id: int, text: str, provider=None
) -> bool:
    """Проверяет, нужно ли предлагать черновик на это входящее сообщение."""
    if not settings.draft_suggestions_enabled:
        return False
    if settings.draft_only_important:
        from src.core.contacts.urgency_classifier import classify_urgency

        urgency = await classify_urgency(text, provider=provider)
        if urgency == "normal":
            return False
    return _check_rate_limit(user_id, settings.draft_max_per_hour)


async def suggest_draft(
    provider: "LLMProvider",
    owner_id: int,
    peer_id: int,
    contact: "Contact",
    incoming_text: str,
    sender_name: str,
    messages: list["Message"],
) -> str | None:
    """Генерирует черновик ответа через LLM."""
    from src.core.intelligence.summarizer import draft_reply

    try:
        draft = await draft_reply(
            provider,
            contact,
            messages,
            heavy=False,
            global_style=None,
            owner_id=owner_id,
        )
        return draft
    except Exception:
        logger.exception("Failed to generate draft for peer %s", peer_id)
        return None
