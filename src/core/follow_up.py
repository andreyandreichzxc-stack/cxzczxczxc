import asyncio
import logging
from datetime import datetime, timedelta, timezone

from src.core.notification_queue import notification_queue
from src.db.models import Notification
from src.db.repo import get_or_create_user, get_contact, list_active_conversations
from src.config import settings
from src.db.session import get_session

logger = logging.getLogger(__name__)


async def follow_up_loop(owner_id: int) -> None:
    """Проверка переписок без ответа >24 часов, раз в 4 часа."""
    while True:
        try:
            async with get_session() as session:
                owner = await get_or_create_user(session, owner_id)
                cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
                convos = await list_active_conversations(
                    session, owner, status="waiting_reply", limit=30
                )
                stale: list[str] = []
                for conv in convos:
                    if conv.last_incoming_at and conv.last_incoming_at < cutoff:
                        if (
                            conv.last_outgoing_at is None
                            or conv.last_outgoing_at < conv.last_incoming_at
                        ):
                            contact = await get_contact(session, owner, conv.peer_id)
                            name = (
                                contact.display_name if contact else str(conv.peer_id)
                            )
                            stale.append(name)

                if stale:
                    names = ", ".join(stale[:5])
                    suffix = f" и ещё {len(stale) - 5}" if len(stale) > 5 else ""
                    await notification_queue.enqueue(
                        topic="follow_up",
                        text=f"⚠️ <b>Без ответа >24ч:</b> {names}{suffix}\n"
                        f"<i>/threads — просмотреть и ответить</i>",
                        priority=Notification.PRIORITY_HIGH,
                    )
            await asyncio.sleep(settings.follow_up_interval_sec)  # раз в 4 часа
        except Exception as e:
            logger.error("FollowUp loop error: %s", e)
            await asyncio.sleep(settings.follow_up_interval_sec)
