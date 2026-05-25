"""3-Level Smart Cache: L0 (in-memory LRU) → L1 (SQLite) → L2 (Memory.fact).

Key features:
- Automatic importance scoring (no user input required)
- Anti-bloat safety: size caps, graduation rate limits, dedup, score decay
- Async-safe: per-key asyncio.Lock to prevent races
- Owner-scoped: L0 keys are composite ``{owner_id}:{key}``
- L0 is per-instance (tests can create fresh instances)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select

from src.db.models._cache import SmartCacheEntry
from src.db.repo import add_memory, get_or_create_user
from src.db.session import get_session

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────

L0_MAX_SIZE = 500
L1_MAX_SIZE_PER_OWNER = 10_000
GRADUATION_MAX_PER_OWNER_PER_DAY = 50
ACCESS_COUNT_CAP = 100
IMPORTANCE_DECAY_PER_HOUR = 0.95
GLOBAL_MAX_GRADUATIONS = 100_000
CLEANUP_EVERY_N_OPS = 1000
STALE_DAYS = 7

SOURCE_WEIGHTS: dict[str, float] = {
    "maestro_synthesis": 0.9,
    "full_analyzer": 0.8,
    "memory_recall": 0.7,
    "agent_result": 0.5,
    "prefetch": 0.2,
}


# ─── SmartCache ───────────────────────────────────────────────────────


class SmartCache:
    """3-level cache with automatic promotion and anti-bloat measures.

    L0 is per-instance (OrderedDict, max 500). L1 is SQLite (shared).
    L2 is Memory.fact (via add_memory).
    """

    # SQLite stores naive datetimes even with DateTime(timezone=True) — helper:
    @staticmethod
    def _ensure_aware(dt: datetime) -> datetime:
        """Return a timezone-aware datetime (UTC)."""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    _MAX_KEY_LOCKS: int = 500  # cap to prevent unbounded growth

    def __init__(self) -> None:
        self._l0: OrderedDict[str, str] = OrderedDict()
        self._l0_lock = asyncio.Lock()
        self._key_locks: dict[str, asyncio.Lock] = {}
        self._ops_since_cleanup: int = 0
        self._cleanup_lock = asyncio.Lock()

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _l0_key(key: str, owner_id: int) -> str:
        """Composite L0 key scoped by owner."""
        return f"{owner_id}:{key}"

    @staticmethod
    def _lock_key(key: str, owner_id: int) -> str:
        """Unique lock key combining owner + cache key."""
        return f"{owner_id}:{key}"

    def _ensure_lock(self, key: str, owner_id: int) -> asyncio.Lock:
        """Return or create a per-(owner,key) asyncio.Lock.

        If _key_locks exceeds _MAX_KEY_LOCKS, evict unlocked entries first.
        """
        lk = self._lock_key(key, owner_id)
        if lk not in self._key_locks:
            # Evict if over cap
            if len(self._key_locks) >= self._MAX_KEY_LOCKS:
                to_remove = [k for k, v in self._key_locks.items() if not v.locked()]
                for k in to_remove[: self._MAX_KEY_LOCKS // 4]:
                    del self._key_locks[k]
            self._key_locks[lk] = asyncio.Lock()
        return self._key_locks[lk]

    # ── Public API ────────────────────────────────────────────────────

    async def get(self, key: str, owner_id: int) -> str | None:
        """Retrieve a value from the best available cache level.

        Order: L0 → L1 → (L2 is handled separately via memory recall).
        L0 hits also synchronously bump L1 access_count for correct scoring.
        On L1 hit, value is promoted to L0 and graduation is evaluated.
        """
        l0k = self._l0_key(key, owner_id)
        value: str | None = None

        # ── L0 check ──
        async with self._l0_lock:
            if l0k in self._l0:
                self._l0.move_to_end(l0k)  # MRU bump
                value = self._l0[l0k]

        if value is not None:
            # L0 hit — synchronously bump L1 access_count for correct tracking
            await self._bump_l1_access(key, owner_id)
            self._ops_since_cleanup += 1
            await self._maybe_cleanup_standalone()
            self._maybe_cleanup_key_locks()
            return value

        # ── L1 check ──
        async with self._ensure_lock(key, owner_id):
            async with get_session() as session:
                result = await session.execute(
                    select(SmartCacheEntry).where(
                        SmartCacheEntry.cache_key == key,
                        SmartCacheEntry.owner_id == owner_id,
                    )
                )
                entry = result.scalar_one_or_none()

                if entry is not None:
                    now = datetime.now(timezone.utc)

                    # Calculate importance BEFORE updating accessed_at
                    # (so decay uses the old timestamp)
                    new_score = self._calculate_importance(entry, entry.source, now)
                    entry.importance_score = new_score
                    entry.access_count = min(entry.access_count + 1, ACCESS_COUNT_CAP)
                    entry.accessed_at = now

                    await session.flush()

                    # Promote to L0
                    async with self._l0_lock:
                        self._enforce_l0_limit()
                        self._l0[l0k] = entry.cache_value
                        self._l0.move_to_end(l0k)

                    # Evaluate graduation to L2
                    if entry.importance_score > 0.7 and not entry.graduated:
                        can_graduate = await self._check_daily_graduation_limit(
                            session, owner_id
                        )
                        if can_graduate:
                            await self._graduate(entry, owner_id, session)

                    self._ops_since_cleanup += 1
                    await self._maybe_cleanup(session)
                    self._maybe_cleanup_key_locks()

                    return entry.cache_value

        # ── Cache miss ──
        self._ops_since_cleanup += 1
        await self._maybe_cleanup_standalone()
        self._maybe_cleanup_key_locks()
        return None

    async def set(
        self,
        key: str,
        value: str,
        owner_id: int,
        source: str = "unknown",
        importance_hint: float = 0.0,
    ) -> None:
        """Store a value across all cache levels.

        Always stored in L0 and L1. If importance_hint > 0.8, graduates
        immediately to L2 (respecting daily limit).
        """
        content_hash = hashlib.sha256(value.encode("utf-8")).hexdigest()
        now = datetime.now(timezone.utc)
        l0k = self._l0_key(key, owner_id)

        async with self._ensure_lock(key, owner_id):
            # ── L0 store ──
            async with self._l0_lock:
                self._enforce_l0_limit()
                self._l0[l0k] = value
                self._l0.move_to_end(l0k)

            # ── L1 store ──
            async with get_session() as session:
                result = await session.execute(
                    select(SmartCacheEntry).where(
                        SmartCacheEntry.cache_key == key,
                        SmartCacheEntry.owner_id == owner_id,
                    )
                )
                existing = result.scalar_one_or_none()

                initial_score = max(
                    self._calculate_importance_raw(
                        access_count=1,
                        created_at=now,
                        accessed_at=now,
                        source=source,
                    ),
                    importance_hint,
                )

                if existing is not None:
                    existing.cache_value = value
                    existing.source = source
                    existing.accessed_at = now
                    existing.access_count = min(
                        existing.access_count + 1, ACCESS_COUNT_CAP
                    )
                    existing.importance_score = initial_score
                    if content_hash != existing.content_hash:
                        existing.graduated = False  # content changed → re-graduate
                    existing.content_hash = content_hash
                    await session.flush()
                    entry = existing
                else:
                    await self._enforce_l1_limit(session, owner_id)
                    entry = SmartCacheEntry(
                        cache_key=key,
                        cache_value=value,
                        source=source,
                        owner_id=owner_id,
                        created_at=now,
                        accessed_at=now,
                        access_count=1,
                        importance_score=initial_score,
                        graduated=False,
                        content_hash=content_hash,
                    )
                    session.add(entry)
                    await session.flush()

                # ── Immediate graduation if high importance ──
                if importance_hint > 0.8 and not entry.graduated:
                    can_graduate = await self._check_daily_graduation_limit(
                        session, owner_id
                    )
                    if can_graduate:
                        if not await self._is_hash_already_graduated(
                            session, content_hash, owner_id
                        ):
                            await self._graduate(entry, owner_id, session)

                await self._maybe_cleanup(session)
                self._maybe_cleanup_key_locks()

        self._ops_since_cleanup += 1

    # ── L0‑hit → L1 sync bump ─────────────────────────────────────────

    async def _bump_l1_access(self, key: str, owner_id: int) -> None:
        """Synchronously bump L1 access_count and recalc importance for L0 hit."""
        try:
            async with self._ensure_lock(key, owner_id):
                async with get_session() as session:
                    result = await session.execute(
                        select(SmartCacheEntry).where(
                            SmartCacheEntry.cache_key == key,
                            SmartCacheEntry.owner_id == owner_id,
                        )
                    )
                    entry = result.scalar_one_or_none()
                    if entry is not None:
                        now = datetime.now(timezone.utc)
                        # Calculate importance BEFORE updating timestamps
                        new_score = self._calculate_importance(entry, entry.source, now)
                        entry.importance_score = new_score
                        entry.access_count = min(
                            entry.access_count + 1, ACCESS_COUNT_CAP
                        )
                        entry.accessed_at = now
                        await session.flush()
        except Exception:
            logger.debug("SmartCache: failed to bump L1 for key=%s", key, exc_info=True)

    # ── Cleanup helpers for paths without a session ────────────────────

    async def _maybe_cleanup_standalone(self) -> None:
        """Trigger cleanup when not inside a DB session."""
        if self._ops_since_cleanup < CLEANUP_EVERY_N_OPS:
            return
        async with self._cleanup_lock:
            if self._ops_since_cleanup >= CLEANUP_EVERY_N_OPS:
                self._ops_since_cleanup = 0
                async with get_session() as session:
                    await self._cleanup_stale_entries(session)

    # ── Importance scoring (automatic) ────────────────────────────────

    @staticmethod
    def _calculate_importance_raw(
        access_count: int,
        created_at: datetime,
        accessed_at: datetime,
        source: str,
    ) -> float:
        """Pure‑function version for scoring without a DB entry."""
        now = datetime.now(timezone.utc)
        created_at = SmartCache._ensure_aware(created_at)
        accessed_at = SmartCache._ensure_aware(accessed_at)
        hours_existed = max((now - created_at).total_seconds() / 3600, 0.01)
        access_rate = min(access_count / hours_existed, 100)

        # 60% — access frequency
        score = 0.6 * min(access_rate / 5.0, 1.0)

        # 25% — source weight
        score += 0.25 * SOURCE_WEIGHTS.get(source, 0.3)

        # 5% — recency bias (fresh data slightly prioritized)
        hours_since_access = (now - accessed_at).total_seconds() / 3600
        score += 0.05 * max(0.0, 1.0 - hours_since_access / 168.0)  # 7‑day window

        return min(score, 1.0)

    @classmethod
    def _calculate_importance(
        cls,
        entry: SmartCacheEntry,
        source: str,
        now: datetime | None = None,
    ) -> float:
        """Calculate importance from a DB entry.

        IMPORTANT: call this BEFORE updating entry.accessed_at, so that
        hours_since_access reflects the actual time since last real access.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        # SQLite returns naive datetimes — make them aware for safe arithmetic
        accessed_at = (
            cls._ensure_aware(entry.accessed_at) if entry.accessed_at else None
        )

        hours_since_access = (
            (now - accessed_at).total_seconds() / 3600 if accessed_at else 999
        )

        base_score = cls._calculate_importance_raw(
            access_count=entry.access_count,
            created_at=entry.created_at,
            accessed_at=entry.accessed_at,
            source=source,
        )

        # Apply natural decay: score *= 0.95 ** hours_since_access
        if hours_since_access > 1:
            decay = IMPORTANCE_DECAY_PER_HOUR**hours_since_access
            base_score *= decay

        return min(base_score, 1.0)

    # ── L0 LRU enforcement ────────────────────────────────────────────

    def _enforce_l0_limit(self) -> None:
        """Evict oldest item if L0 exceeds max size."""
        while len(self._l0) > L0_MAX_SIZE:
            self._l0.popitem(last=False)

    # ── L1 size enforcement ───────────────────────────────────────────

    async def _enforce_l1_limit(self, session: Any, owner_id: int) -> None:
        """Delete oldest non-graduated entries if per-owner L1 limit exceeded."""
        count_result = await session.execute(
            select(func.count()).where(SmartCacheEntry.owner_id == owner_id)
        )
        current_count = count_result.scalar_one()

        if current_count >= L1_MAX_SIZE_PER_OWNER:
            excess = current_count - L1_MAX_SIZE_PER_OWNER + 1
            old = await session.execute(
                select(SmartCacheEntry)
                .where(
                    SmartCacheEntry.owner_id == owner_id,
                    SmartCacheEntry.graduated.is_(False),
                )
                .order_by(SmartCacheEntry.accessed_at.asc())
                .limit(max(excess, 1))
            )
            for entry in old.scalars().all():
                l0k = self._l0_key(entry.cache_key, owner_id)
                async with self._l0_lock:
                    self._l0.pop(l0k, None)
                await session.delete(entry)

    # ── Graduation to L2 ──────────────────────────────────────────────

    async def _check_daily_graduation_limit(
        self,
        session: Any,
        owner_id: int,
    ) -> bool:
        """Check global cap and per-owner daily graduation limit."""
        global_result = await session.execute(
            select(func.count()).where(SmartCacheEntry.graduated.is_(True))
        )
        global_count = global_result.scalar_one()
        if global_count >= GLOBAL_MAX_GRADUATIONS:
            logger.critical(
                "SmartCache: global graduation cap %d reached.",
                GLOBAL_MAX_GRADUATIONS,
            )
            return False

        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        daily_result = await session.execute(
            select(func.count()).where(
                SmartCacheEntry.owner_id == owner_id,
                SmartCacheEntry.graduated.is_(True),
                SmartCacheEntry.created_at >= today_start,
            )
        )
        daily_count = daily_result.scalar_one()
        if daily_count >= GRADUATION_MAX_PER_OWNER_PER_DAY:
            logger.debug(
                "SmartCache: daily graduation limit %d reached for owner %d",
                GRADUATION_MAX_PER_OWNER_PER_DAY,
                owner_id,
            )
            return False

        return True

    async def _is_hash_already_graduated(
        self,
        session: Any,
        content_hash: str,
        owner_id: int,
    ) -> bool:
        """Check if any graduated entry with the same content_hash exists."""
        if not content_hash:
            return False
        result = await session.execute(
            select(func.count()).where(
                SmartCacheEntry.content_hash == content_hash,
                SmartCacheEntry.owner_id == owner_id,
                SmartCacheEntry.graduated.is_(True),
            )
        )
        return result.scalar_one() > 0

    async def _graduate(
        self,
        entry: SmartCacheEntry,
        owner_id: int,
        session: Any,
    ) -> None:
        """Promote a cache entry to L2 (Memory.fact) via add_memory()."""
        if entry.graduated:
            return

        if entry.content_hash and await self._is_hash_already_graduated(
            session, entry.content_hash, owner_id
        ):
            logger.debug(
                "SmartCache: skipping graduation — content hash already graduated"
            )
            entry.graduated = True
            await session.flush()
            return

        fact = f"[{entry.source}] {entry.cache_value[:500]}"

        owner = await get_or_create_user(session, owner_id)
        await add_memory(
            session,
            owner,
            fact=fact,
            source=entry.source,
            memory_type="cached_knowledge",
            memory_tier=3,
            importance=min(entry.importance_score, 1.0),
            confidence=min(entry.importance_score, 0.95),
            deduplicate=True,
        )

        entry.graduated = True
        entry.accessed_at = datetime.now(timezone.utc)
        await session.flush()
        logger.info(
            "SmartCache: graduated key=%s source=%s score=%.3f",
            entry.cache_key,
            entry.source,
            entry.importance_score,
        )

    # ── Key-lock cleanup (prevent _key_locks leak) ────────────────────

    def _maybe_cleanup_key_locks(self) -> None:
        """Remove unused locks from _key_locks every 1000 ops."""
        if self._ops_since_cleanup < 1000:
            return
        self._ops_since_cleanup = 0
        unused = [k for k, v in self._key_locks.items() if not v.locked()]
        for k in unused:
            del self._key_locks[k]

    # ── Periodic cleanup ──────────────────────────────────────────────

    async def _maybe_cleanup(self, session: Any) -> None:
        """Run cleanup every N operations (called from within a session)."""
        if self._ops_since_cleanup < CLEANUP_EVERY_N_OPS:
            return
        self._ops_since_cleanup = 0
        await self._cleanup_stale_entries(session)

    async def _cleanup_stale_entries(self, session: Any) -> None:
        """Delete L1 entries: access_count=0, older than 7 days, not graduated."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=STALE_DAYS)
        result = await session.execute(
            select(SmartCacheEntry).where(
                SmartCacheEntry.access_count == 0,
                SmartCacheEntry.created_at < cutoff,
                SmartCacheEntry.graduated.is_(False),
            )
        )
        stale = result.scalars().all()
        for entry in stale:
            l0k = self._l0_key(entry.cache_key, entry.owner_id)
            async with self._l0_lock:
                self._l0.pop(l0k, None)
            await session.delete(entry)
        if stale:
            logger.debug("SmartCache: cleaned up %d stale entries", len(stale))
            await session.flush()


# ── Module-level singleton ────────────────────────────────────────────

smart_cache = SmartCache()
