"""Кэш memory-статистики. TTL = 5 минут, инвалидация при изменении."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CacheEntry:
    data: Any
    timestamp: float = field(default_factory=time.time)
    ttl: float = 300.0  # 5 минут

    def is_valid(self) -> bool:
        return (time.time() - self.timestamp) < self.ttl


_stats: dict[str, CacheEntry] = {}
_lock = asyncio.Lock()


async def get_cached(key: str) -> Any | None:
    async with _lock:
        entry = _stats.get(key)
        if entry:
            if entry.is_valid():
                return entry.data
            # Evict expired entry
            del _stats[key]
        return None


async def set_cache(key: str, data: Any, ttl: float = 300.0) -> None:
    async with _lock:
        _stats[key] = CacheEntry(data=data, timestamp=time.time(), ttl=ttl)


async def invalidate(prefix: str = "") -> None:
    """Инвалидировать кэш. Если prefix пустой — всё. Иначе по префиксу."""
    async with _lock:
        if not prefix:
            _stats.clear()
        else:
            keys_to_del = [k for k in _stats if k.startswith(prefix)]
            for k in keys_to_del:
                del _stats[k]
