"""CRUD для отложенных сообщений."""

from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models._messaging import ScheduledMessage


async def create_scheduled(
    session: AsyncSession,
    user_id: int,
    contact_name: str,
    text: str,
    send_at: datetime,
) -> ScheduledMessage:
    msg = ScheduledMessage(
        user_id=user_id,
        contact_name=contact_name,
        text=text,
        send_at=send_at,
    )
    session.add(msg)
    await session.flush()
    return msg


async def get_pending(session: AsyncSession) -> list[ScheduledMessage]:
    """Возвращает все pending сообщения где send_at <= now (UTC)."""
    now = datetime.now(timezone.utc)
    result = await session.execute(
        select(ScheduledMessage)
        .where(ScheduledMessage.status == "pending")
        .where(ScheduledMessage.send_at <= now)
        .order_by(ScheduledMessage.send_at)
    )
    return list(result.scalars().all())


async def mark_sent(session: AsyncSession, msg_id: int) -> None:
    await session.execute(
        update(ScheduledMessage)
        .where(ScheduledMessage.id == msg_id)
        .values(status="sent", sent_at=datetime.now(timezone.utc))
    )


async def mark_failed(session: AsyncSession, msg_id: int, error: str) -> None:
    await session.execute(
        update(ScheduledMessage)
        .where(ScheduledMessage.id == msg_id)
        .values(status="failed", error=error[:500])
    )


async def get_user_pending(
    session: AsyncSession,
    user_id: int,
) -> list[ScheduledMessage]:
    result = await session.execute(
        select(ScheduledMessage)
        .where(ScheduledMessage.user_id == user_id)
        .where(ScheduledMessage.status == "pending")
        .order_by(ScheduledMessage.send_at)
    )
    return list(result.scalars().all())
