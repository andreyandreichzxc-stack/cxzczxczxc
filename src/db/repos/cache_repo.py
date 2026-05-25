"""Cache repository — AgentCache."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import AgentCache

logger = logging.getLogger(__name__)


async def get_agent_cache(session: AsyncSession, cache_key: str) -> str | None:
    """Получить кэш агента."""
    result = await session.execute(
        select(AgentCache).where(AgentCache.cache_key == cache_key)
    )
    row = result.scalar_one_or_none()
    if row:
        now = datetime.now(timezone.utc)
        # Handle both old naive and new aware datetimes
        try:
            age = (now - row.created_at).total_seconds()
        except TypeError:
            from datetime import timezone as tz

            age = (now - row.created_at.replace(tzinfo=tz.utc)).total_seconds()
        if age < row.ttl_seconds:
            return row.result_json
        await session.delete(row)
        await session.flush()
    return None


async def upsert_agent_cache(
    session: AsyncSession, cache_key: str, result_json: str, ttl_seconds: int
) -> None:
    """Сохранить/обновить кэш агента."""
    result = await session.execute(
        select(AgentCache).where(AgentCache.cache_key == cache_key)
    )
    row = result.scalar_one_or_none()
    if row:
        row.result_json = result_json
        row.created_at = datetime.now(timezone.utc)
        row.ttl_seconds = ttl_seconds
    else:
        session.add(
            AgentCache(
                cache_key=cache_key,
                result_json=result_json,
                ttl_seconds=ttl_seconds,
            )
        )
    await session.flush()
