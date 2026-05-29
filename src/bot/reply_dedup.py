"""Reply deduplication — prevents sending the same text twice to the same chat."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from hashlib import sha256
import time


class ReplyDedup:
    """Hash-based cache with TTL to prevent duplicate replies."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600) -> None:
        self._cache: OrderedDict[str, float] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds

    async def is_duplicate(self, chat_id: int, text: str) -> bool:
        loop = asyncio.get_running_loop()
        digest = await loop.run_in_executor(
            None,
            lambda: sha256(text.encode(), usedforsecurity=False).hexdigest()[:16],
        )
        key = f"{chat_id}:{digest}"
        now = time.monotonic()
        # Evict stale entries
        while self._cache and next(iter(self._cache.values())) < now - self._ttl:
            self._cache.popitem(last=False)
        if key in self._cache:
            return True
        # Evict oldest if at capacity
        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        self._cache[key] = now
        return False


dedup = ReplyDedup()
