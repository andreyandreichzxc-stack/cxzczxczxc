"""Двухуровневый кэш для результатов агентов: in-memory LRU + SQLite."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import OrderedDict
from typing import Any

from src.db.session import get_session

logger = logging.getLogger(__name__)

# In-memory LRU (макс 200 записей)
_memory_cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
_MAX_MEMORY = 200

# Per-key locks to prevent cache stampede on cold misses
_key_locks: dict[str, asyncio.Lock] = {}


async def _get_lock(key: str) -> asyncio.Lock:
    if key not in _key_locks:
        _key_locks[key] = asyncio.Lock()
    return _key_locks[key]


def _cache_key(agent_type: str, params_hash: str) -> str:
    return f"{agent_type}:{params_hash}"


def cache_get(agent_type: str, params_hash: str) -> Any | None:
    """Достать из in-memory кэша. Возвращает None если нет или протух."""
    key = _cache_key(agent_type, params_hash)
    if key in _memory_cache:
        expires_at, value = _memory_cache[key]
        if time.time() < expires_at:
            _memory_cache.move_to_end(key)  # LRU: обновить позицию
            return value
        del _memory_cache[key]
    return None


def cache_set(agent_type: str, params_hash: str, value: Any, ttl_seconds: int) -> None:
    """Сохранить в in-memory кэш."""
    key = _cache_key(agent_type, params_hash)
    _memory_cache[key] = (time.time() + ttl_seconds, value)
    _memory_cache.move_to_end(key)
    # Вытеснить старые если превышен лимит
    while len(_memory_cache) > _MAX_MEMORY:
        _memory_cache.popitem(last=False)


async def cache_get_db(agent_type: str, params_hash: str) -> Any | None:
    """Достать из SQLite кэша."""
    from src.db.repo import get_agent_cache

    async with get_session() as session:
        row = await get_agent_cache(session, f"{agent_type}:{params_hash}")
        if row:
            try:
                return json.loads(row)
            except Exception:
                return None
    return None


async def cache_set_db(
    agent_type: str, params_hash: str, value: Any, ttl_seconds: int
) -> None:
    """Сохранить в SQLite кэш."""
    from src.db.repo import upsert_agent_cache

    async with get_session() as session:
        await upsert_agent_cache(
            session,
            cache_key=f"{agent_type}:{params_hash}",
            result_json=json.dumps(value, ensure_ascii=False),
            ttl_seconds=ttl_seconds,
        )


async def cache_get_or_set(
    agent_type: str,
    params_hash: str,
    factory,  # async callable
    ttl_seconds: int = 0,
) -> Any:
    """Трёхуровневый доступ: memory → SQLite → factory."""
    if ttl_seconds <= 0:
        return await factory()

    key = _cache_key(agent_type, params_hash)

    # 1. In-memory (fast path — no lock needed for reads)
    val = cache_get(agent_type, params_hash)
    if val is not None:
        return val

    async with await _get_lock(key):
        # Double-check after acquiring per-key lock (stampede guard)
        val = cache_get(agent_type, params_hash)
        if val is not None:
            return val

        # 2. SQLite
        val = await cache_get_db(agent_type, params_hash)
        if val is not None:
            cache_set(agent_type, params_hash, val, ttl_seconds)
            return val

        # 3. Вычислить
        val = await factory()
        if val is not None:
            cache_set(agent_type, params_hash, val, ttl_seconds)
            try:
                await cache_set_db(agent_type, params_hash, val, ttl_seconds)
            except Exception:
                logger.debug(
                    "agent_cache persist failed: %s:%s",
                    agent_type,
                    params_hash,
                    exc_info=True,
                )
                # In-memory кэша достаточно для не-JSON-сериализуемых типов
                pass
        return val
