"""News repository — NewsTopic."""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import (
    NewsTopic,
)

logger = logging.getLogger(__name__)


async def list_news_topics(
    session: AsyncSession,
    user,
    *,
    only_enabled: bool = False,
) -> list[NewsTopic]:
    query = (
        select(NewsTopic)
        .where(NewsTopic.user_id == user.id)
        .order_by(NewsTopic.created_at.asc())
    )
    if only_enabled:
        query = query.where(NewsTopic.enabled.is_(True))
    result = await session.execute(query)
    return list(result.scalars().all())


async def add_news_topic(
    session: AsyncSession,
    user,
    topic: str,
    *,
    hours: int = 24,
) -> NewsTopic:
    nt = NewsTopic(user_id=user.id, topic=topic.strip(), hours=hours)
    session.add(nt)
    await session.flush()
    return nt


async def delete_news_topic(session: AsyncSession, user, topic_id: int) -> bool:
    nt = await session.get(NewsTopic, topic_id)
    if nt is None or nt.user_id != user.id:
        return False
    await session.delete(nt)
    await session.flush()
    return True


async def toggle_news_topic(session: AsyncSession, user, topic_id: int) -> bool | None:
    nt = await session.get(NewsTopic, topic_id)
    if nt is None or nt.user_id != user.id:
        return None
    nt.enabled = not nt.enabled
    await session.flush()
    return nt.enabled
