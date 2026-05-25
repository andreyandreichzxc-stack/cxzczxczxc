"""Burnout detector — notices when user is overwhelmed and offers help."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from functools import partial

from sqlalchemy import desc, select

from src.config import settings
from src.core.infra.task_manager import task_manager
from src.core.scheduling.notification_queue import notification_queue
from src.db.models import Notification
from src.db.models._messaging import Message
from src.db.repo import get_contact, get_or_create_user, list_active_conversations
from src.db.session import get_session

logger = logging.getLogger(__name__)


async def check_burnout(owner_telegram_id: int) -> str | None:
    """Check if the owner shows signs of burnout.

    Returns:
        None if everything is fine.
        str — a supportive message with suggested actions.
    """
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_telegram_id)
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        three_days_ago = now - timedelta(days=3)

        # 1. Check recent outgoing messages for patterns
        recent_out = await session.execute(
            select(Message)
            .where(
                Message.user_id == owner.id,
                Message.is_outgoing == True,
                Message.date >= three_days_ago,
            )
            .order_by(desc(Message.date))
            .limit(30)
        )
        outgoing = recent_out.scalars().all()

        if len(outgoing) < 5:
            return None  # not enough data

        # 2. Count short/dismissive responses
        short_count = 0
        dry_words = {
            "ок",
            "окей",
            "да",
            "нет",
            "ага",
            "понял",
            "принял",
            "ясно",
            "хорошо",
            "ладно",
            "посмотрю",
            "позже",
            "потом",
        }
        for msg in outgoing:
            text = (msg.text or "").lower().strip()
            if len(text) < 20 and any(w in text for w in dry_words):
                short_count += 1

        short_ratio = short_count / len(outgoing) if outgoing else 0

        # 3. Count unreplied messages
        convs = await list_active_conversations(session, owner, limit=50)
        unreplied = 0
        unreplied_names: list[str] = []
        for c in convs:
            if c.last_incoming_at and (
                not c.last_outgoing_at or c.last_incoming_at > c.last_outgoing_at
            ):
                hours = (now - c.last_incoming_at).total_seconds() / 3600
                if hours > 24:
                    unreplied += 1
                    contact = await get_contact(session, owner, c.peer_id)
                    name = contact.display_name if contact else str(c.peer_id)
                    unreplied_names.append(name)

        # 4. Decision
        is_burnout = short_ratio > 0.6 and unreplied >= 2

        if not is_burnout:
            return None

        # Build message
        names = unreplied_names[:3]
        contact_str = ", ".join(names) if names else "никому"

        return (
            f"💭 Слушай, ты последние дни отвечаешь коротко "
            f"(«ок», «да», «потом» — {short_count}/{len(outgoing)} сообщений), "
            f"и не ответил {contact_str}.\n\n"
            f"Всё норм? Если завал — могу сам ответить что ты занят. Скажи кому и что."
        )


async def burnout_loop(owner_telegram_id: int) -> None:
    """Periodic burnout check loop — runs every 6 hours."""
    while True:
        try:
            msg = await check_burnout(owner_telegram_id)
            if msg:
                await notification_queue.enqueue(
                    topic="wellness",
                    text=msg,
                    priority=Notification.PRIORITY_HIGH,
                )
                logger.info("Burnout detected, notification sent")
        except Exception:
            logger.debug("Burnout check failed", exc_info=True)
        await asyncio.sleep(21600)  # every 6 hours


task_manager.register(
    "burnout-checker",
    partial(burnout_loop, settings.owner_telegram_id),
)
