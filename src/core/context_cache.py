"""In-memory TTL-кэш с инвалидацией по префиксу.

Синглтон на уровне модуля. Все операции синхронные.
TTL отсчитывается через time.monotonic().
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)

MAX_CACHE_SIZE = 2000
"""Максимальное количество записей в кэше. При превышении удаляется самая старая запись."""

_cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
"""Кэш: ключ -> (expires_at_monotonic, value). OrderedDict для O(1) eviction."""

_cache_lock = asyncio.Lock()
"""Async lock for _cache — prevents TOCTOU races in async code."""


async def get(key: str) -> Any | None:
    """Вернуть значение из кэша или None если ключа нет / TTL истёк."""
    async with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return None

        expires_at, value = entry
        if time.monotonic() < expires_at:
            return value

        # TTL истёк — удалить (TOCTOU-safe: pop instead of check-then-del)
        _cache.pop(key, None)
        logger.debug("Cache expired: %s", key)
        return None


async def put(key: str, value: Any, ttl: int = 30) -> None:
    """Сохранить значение в кэш с TTL в секундах (по умолчанию 30)."""
    async with _cache_lock:
        if len(_cache) >= MAX_CACHE_SIZE:
            # Evict oldest entry (first in OrderedDict) — O(1)
            _cache.popitem(last=False)
            logger.debug("Cache evicted oldest (OrderedDict popitem)")

        expires_at = time.monotonic() + ttl
        _cache[key] = (expires_at, value)
        logger.debug("Cache put: %s (ttl=%ds)", key, ttl)


async def invalidate(prefix: str = "") -> None:
    """Очистить кэш.

    Если prefix пуст — очистить весь кэш.
    Иначе — удалить только ключи, начинающиеся с prefix.
    """
    async with _cache_lock:
        if not prefix:
            _cache.clear()
            logger.debug("Cache fully invalidated")
            return

        keys_to_del = [k for k in _cache if k.startswith(prefix)]
        for k in keys_to_del:
            del _cache[k]

        if keys_to_del:
            logger.debug(
                "Cache invalidated: prefix=%s (%d keys)", prefix, len(keys_to_del)
            )
