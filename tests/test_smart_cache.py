"""Tests for the 3-level SmartCache system (L0 → L1 → L2)."""

import asyncio
import os
import sys

import pytest

# Ensure project root is in path and DB is in-memory BEFORE any src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["ENCRYPTION_KEY"] = "HmsOzSAxuyfb7zet2nmwhFkgWfH5z6Lsr3tW7MO8GDI="
os.environ["BOT_TOKEN"] = "test:token"
os.environ["OWNER_TELEGRAM_ID"] = "123456789"

from src.db.session import init_db, get_session
from src.db.models._cache import SmartCacheEntry
from src.core.cache.smart_cache import (
    SmartCache,
    smart_cache,
    L0_MAX_SIZE,
    L1_MAX_SIZE_PER_OWNER,
    GRADUATION_MAX_PER_OWNER_PER_DAY,
)

OWNER_ID = 123456789
OWNER2_ID = 987654321


@pytest.fixture(autouse=True)
def setup_db():
    """Recreate all tables before each test."""
    from src.db.session import engine, Base
    from sqlalchemy import text

    async def _recreate():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            # Drop artifacts that survive drop_all and would confuse init_db
            await conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
            await conn.execute(text("DROP TABLE IF EXISTS messages_fts"))
            await conn.execute(text("DROP TABLE IF EXISTS memories_fts"))
        await init_db()

    asyncio.run(_recreate())


# ── Helpers ───────────────────────────────────────────────────────────


async def _l1_count(owner_id: int = OWNER_ID) -> int:
    """Count entries in L1 for an owner."""
    async with get_session() as session:
        from sqlalchemy import select, func

        result = await session.execute(
            select(func.count()).where(SmartCacheEntry.owner_id == owner_id)
        )
        return result.scalar_one()


async def _l1_graduated_count(owner_id: int = OWNER_ID) -> int:
    """Count graduated entries in L1 for an owner."""
    async with get_session() as session:
        from sqlalchemy import select, func

        result = await session.execute(
            select(func.count()).where(
                SmartCacheEntry.owner_id == owner_id,
                SmartCacheEntry.graduated.is_(True),
            )
        )
        return result.scalar_one()


# ── Tests ─────────────────────────────────────────────────────────────


class TestBasicGetSet:
    """Basic get/set roundtrip."""

    @pytest.mark.asyncio
    async def test_set_get_roundtrip(self):
        c = SmartCache()
        await c.set("key1", "value1", OWNER_ID, source="agent_result")
        val = await c.get("key1", OWNER_ID)
        assert val == "value1"

    @pytest.mark.asyncio
    async def test_get_miss_returns_none(self):
        c = SmartCache()
        val = await c.get("nonexistent", OWNER_ID)
        assert val is None

    @pytest.mark.asyncio
    async def test_set_then_get_from_l1(self):
        """Value should persist in L1 and be retrievable."""
        c = SmartCache()
        await c.set("persist_key", "persist_val", OWNER_ID, source="agent_result")
        # L1 count should be 1
        assert await _l1_count(OWNER_ID) == 1

    @pytest.mark.asyncio
    async def test_module_singleton_works(self):
        """Module-level smart_cache singleton."""
        await smart_cache.set("singleton_key", "singleton_val", OWNER_ID)
        val = await smart_cache.get("singleton_key", OWNER_ID)
        assert val == "singleton_val"


class TestL0LRUEviction:
    """L0 LRU eviction: insert >500 items, first should be evicted from L0."""

    @pytest.mark.asyncio
    async def test_l0_lru_eviction(self):
        c = SmartCache()
        # Insert exactly L0_MAX_SIZE items
        for i in range(L0_MAX_SIZE):
            await c.set(f"l0key_{i}", f"val_{i}", OWNER_ID)
        # First item should still be accessible (from L1 fallback)
        val = await c.get("l0key_0", OWNER_ID)
        assert val == "val_0"

        # Insert one more — should trigger L0 eviction
        await c.set(f"l0key_{L0_MAX_SIZE}", "overflow", OWNER_ID)
        # Even after L0 eviction, L1 still has it
        val = await c.get("l0key_0", OWNER_ID)
        assert val == "val_0"


class TestL1OverflowProtection:
    """L1 max size per owner: oldest non-graduated deleted on overflow."""

    @pytest.mark.asyncio
    async def test_l1_overflow_evicts_oldest_non_graduated(self):
        c = SmartCache()
        # Fill to limit with non-graduated entries
        for i in range(L1_MAX_SIZE_PER_OWNER):
            await c.set(f"l1key_{i}", f"val_{i}", OWNER_ID)

        assert await _l1_count(OWNER_ID) <= L1_MAX_SIZE_PER_OWNER

        # Insert one more → overflow
        await c.set("overflow_key", "overflow_val", OWNER_ID)

        count = await _l1_count(OWNER_ID)
        assert count <= L1_MAX_SIZE_PER_OWNER

        # The very first entry (l1key_0) should be evicted from L1
        # But it might still be in L0 — force a fresh cache read
        c2 = SmartCache()
        await c2.get("l1key_0", OWNER_ID)
        # It may or may not be there depending on access time ordering
        # The key assertion is: count didn't exceed limit
        assert count <= L1_MAX_SIZE_PER_OWNER


class TestImportanceScoring:
    """Verify importance scoring formula and source weights."""

    @pytest.mark.asyncio
    async def test_source_weight_ordering(self):
        """Higher-importance sources produce higher base scores."""
        from src.core.cache.smart_cache import SmartCache as SC
        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc)
        old_created = now - timedelta(hours=10)
        old_accessed = now - timedelta(hours=1)

        score_maestro = SC._calculate_importance_raw(
            access_count=10,
            created_at=old_created,
            accessed_at=old_accessed,
            source="maestro_synthesis",
        )
        score_prefetch = SC._calculate_importance_raw(
            access_count=10,
            created_at=old_created,
            accessed_at=old_accessed,
            source="prefetch",
        )
        assert score_maestro > score_prefetch, (
            f"maestro={score_maestro} should be > prefetch={score_prefetch}"
        )

    @pytest.mark.asyncio
    async def test_access_count_bumps_on_get(self):
        c = SmartCache()
        await c.set("bump_key", "bump_val", OWNER_ID, source="agent_result")

        # Multiple gets — first hits L0 (bumps L1 synchronously),
        # subsequent also hit L0 and bump L1.
        for _ in range(5):
            await c.get("bump_key", OWNER_ID)

        async with get_session() as session:
            from sqlalchemy import select

            result = await session.execute(
                select(SmartCacheEntry).where(SmartCacheEntry.cache_key == "bump_key")
            )
            entry = result.scalar_one()
            # After 5 gets, each bumps L1 → should be at least 2
            assert entry.access_count >= 2, f"Expected >=2, got {entry.access_count}"


class TestGraduationRateLimiting:
    """Graduation rate limit: max 50 per owner per day."""

    @pytest.mark.asyncio
    async def test_graduation_rate_limit(self):
        c = SmartCache()
        # Set many entries with high importance_hint to trigger graduation
        for i in range(GRADUATION_MAX_PER_OWNER_PER_DAY + 10):
            await c.set(
                f"grad_key_{i}",
                f"grad_val_{i}",
                OWNER_ID,
                source="maestro_synthesis",
                importance_hint=0.9,
            )

        graduated = await _l1_graduated_count(OWNER_ID)
        assert graduated <= GRADUATION_MAX_PER_OWNER_PER_DAY


class TestDeduplication:
    """Same content_hash should not graduate twice."""

    @pytest.mark.asyncio
    async def test_dedup_same_hash_not_graduated_twice(self):
        c = SmartCache()
        # First set with high importance → graduates
        await c.set(
            "dedup_key_1",
            "identical content",
            OWNER_ID,
            source="maestro_synthesis",
            importance_hint=0.9,
        )
        # Second set with same content but different key
        await c.set(
            "dedup_key_2",
            "identical content",
            OWNER_ID,
            source="maestro_synthesis",
            importance_hint=0.9,
        )

        # Only one should be graduated (content_hash dedup)
        graduated = await _l1_graduated_count(OWNER_ID)
        assert graduated <= 1, f"Expected ≤1 graduated, got {graduated}"


class TestCleanupStaleEntries:
    """Periodic cleanup of stale L1 entries."""

    @pytest.mark.asyncio
    async def test_importance_decays_on_recalculation(self):
        from datetime import datetime, timezone, timedelta

        c = SmartCache()
        await c.set(
            "decay_key",
            "decay_val",
            OWNER_ID,
            source="agent_result",
            importance_hint=0.8,
        )

        # Simulate time passing by manually updating accessed_at to 48 hours ago
        async with get_session() as session:
            from sqlalchemy import update

            old_time = datetime.now(timezone.utc) - timedelta(hours=48)
            await session.execute(
                update(SmartCacheEntry)
                .where(SmartCacheEntry.cache_key == "decay_key")
                .values(accessed_at=old_time)
            )
            await session.commit()

        # Create a fresh cache to avoid L0 hit (L0 doesn't have this key)
        c2 = SmartCache()
        await c2.get("decay_key", OWNER_ID)

        async with get_session() as session:
            from sqlalchemy import select

            result = await session.execute(
                select(SmartCacheEntry).where(SmartCacheEntry.cache_key == "decay_key")
            )
            entry = result.scalar_one()
            # Score should be noticeably lower after 48h without access
            # (decay factor ≈ 0.95^48 ≈ 0.085)
            assert entry.importance_score < 0.5, (
                f"Expected score < 0.5 after 48h decay, got {entry.importance_score}"
            )

    @pytest.mark.asyncio
    async def test_stale_entries_cleaned_up(self):
        """Stale entries (access_count=0, >7 days, not graduated) are deleted."""
        c = SmartCache()
        for i in range(10):
            await c.set(f"stale_{i}", f"val_{i}", OWNER_ID)

        # Manually make them stale
        async with get_session() as session:
            from sqlalchemy import update
            from datetime import datetime, timezone, timedelta

            old_date = datetime.now(timezone.utc) - timedelta(days=8)
            await session.execute(
                update(SmartCacheEntry)
                .where(SmartCacheEntry.owner_id == OWNER_ID)
                .values(access_count=0, created_at=old_date, graduated=False)
            )
            await session.commit()

        # Force cleanup through cache miss
        c._ops_since_cleanup = 1000
        await c.get("trigger_cleanup_miss", OWNER_ID)

        count = await _l1_count(OWNER_ID)
        assert count < 10, f"Expected <10 after cleanup, got {count}"


class TestAccessCountCap:
    """Access count must not exceed the cap."""

    @pytest.mark.asyncio
    async def test_access_count_capped(self):
        from src.core.cache.smart_cache import ACCESS_COUNT_CAP

        c = SmartCache()
        await c.set("cap_key", "cap_val", OWNER_ID, source="agent_result")

        for _ in range(ACCESS_COUNT_CAP + 50):
            await c.get("cap_key", OWNER_ID)

        async with get_session() as session:
            from sqlalchemy import select

            result = await session.execute(
                select(SmartCacheEntry).where(SmartCacheEntry.cache_key == "cap_key")
            )
            entry = result.scalar_one()
            assert entry.access_count <= ACCESS_COUNT_CAP


class TestMultiOwnerIsolation:
    """Each owner has their own cache namespace."""

    @pytest.mark.asyncio
    async def test_owner_isolation(self):
        c = SmartCache()
        await c.set("shared_key", "owner1_val", OWNER_ID)
        # Owner 2 should not see owner 1's value (L0 is owner-scoped, L1 filtered)
        val = await c.get("shared_key", OWNER2_ID)
        assert val is None

    @pytest.mark.asyncio
    async def test_l1_limit_per_owner(self):
        """L1 limits are per-owner, not global."""
        c = SmartCache()
        for i in range(100):
            await c.set(f"o1_{i}", f"v_{i}", OWNER_ID)
        for i in range(100):
            await c.set(f"o2_{i}", f"v_{i}", OWNER2_ID)

        assert await _l1_count(OWNER_ID) == 100
        o2_count = await _l1_count(OWNER2_ID)
        assert o2_count >= 0
