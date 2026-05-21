"""Memory Clusterer — собирает факты в кластеры по темам."""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone

from sqlalchemy import select as _sel

from src.db.models import Memory, MemoryCluster, MemoryClusterMember
from src.db.repo import (
    add_member,
    get_or_create_user,
    list_contacts,
    list_memories,
    upsert_memory_cluster,
)
from src.config import settings
from src.db.session import get_session

logger = logging.getLogger(__name__)

UTC = lambda: datetime.now(timezone.utc).replace(tzinfo=None)


async def rebuild_clusters(telegram_id: int) -> int:
    """Пересобирает кластеры: группирует активные факты по contact_id + topic."""
    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)
        facts = await list_memories(session, owner)
        active = [m for m in facts if m.is_active and m.fact and len(m.fact) > 5]
        if len(active) < 5:
            return 0

        # Стратегия 1: группировка по contact_id + cluster_topic
        groups: dict[tuple, list] = defaultdict(list)
        for m in active:
            key = (m.contact_id or 0, m.cluster_topic or "общее")
            groups[key].append(m)

        created = 0
        for (contact_id, topic), facts in groups.items():
            if len(facts) < 3:
                continue
            # Upsert кластер
            cluster = await upsert_memory_cluster(
                session,
                owner,
                topic,
                summary=f"{len(facts)} фактов"
                + (f" о контакте {contact_id}" if contact_id else ""),
                fact_count=len(facts),
            )
            # Membership
            for m in facts:
                score = (m.confidence or 0.5) * 0.6 + (0.4 if m.pinned else 0)
                await add_member(session, owner.id, m.id, cluster.id, min(score, 1.0))
            created += 1

        await session.flush()
        return created


async def cluster_first_retrieval(
    telegram_id: int, query: str, limit: int = 5
) -> list[Memory]:
    """Cluster-first retrieval: сначала кластеры, потом факты."""
    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)

        # Шаг 1: найти кластеры по topic (ILIKE)
        q = (
            _sel(MemoryCluster)
            .where(
                MemoryCluster.user_id == owner.id,
                MemoryCluster.topic.ilike(f"%{query}%"),
            )
            .limit(3)
        )
        r = await session.execute(q)
        clusters = list(r.scalars().all())

        # Шаг 2: собрать лучшие факты из найденных кластеров
        if not clusters:
            return []

        cluster_ids = [c.id for c in clusters]
        q2 = (
            _sel(Memory, MemoryClusterMember.relevance_score)
            .join(
                MemoryClusterMember,
                Memory.id == MemoryClusterMember.memory_id,
            )
            .where(
                MemoryClusterMember.cluster_id.in_(cluster_ids),
                Memory.is_active == True,
                Memory.user_id == owner.id,
            )
            .order_by(
                MemoryClusterMember.relevance_score.desc(),
                Memory.confidence.desc(),
            )
            .limit(limit)
        )
        r2 = await session.execute(q2)
        return [row[0] for row in r2.all()]


async def cluster_loop(telegram_id: int) -> None:
    """Фоновый цикл: пересборка кластеров ночью (03:00)."""
    from src.core.infra.timeutil import get_user_tz, now_in_tz

    last_run = None
    while True:
        try:
            async with get_session() as session:
                owner = await get_or_create_user(session, telegram_id)
                tz = get_user_tz(owner)
            now = now_in_tz(tz)
            if now.hour == 3 and last_run != now.date():
                last_run = now.date()
                n = await rebuild_clusters(telegram_id)
                logger.info("Clusters rebuilt: %d created", n)
        except Exception:
            logger.exception("Cluster loop error")

        # --- L2 Scene extraction after cluster rebuild ---
        try:
            from src.core.memory.scene_extractor import extract_scenes_for_user

            scenes = await extract_scenes_for_user(telegram_id)
            if scenes:
                logger.info("L2 scenes generated: %d", scenes)
        except Exception:
            logger.debug("Scene extraction skipped (non-critical)", exc_info=True)

        await asyncio.sleep(settings.memory_clusterer_interval_sec)
