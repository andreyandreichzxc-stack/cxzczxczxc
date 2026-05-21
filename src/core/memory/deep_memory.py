"""
Deep memory retrieval: tier 2-3 prefetch + BFS traversal on MemoryLink graph.

Адаптирован под реальную схему БД:
- Memory.memory_tier (не tier)
- MemoryLink.source_id / target_id (не memory_id / related_memory_id)
- Все операции асинхронные (AsyncSession)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy import select, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Memory, MemoryLink

logger = logging.getLogger(__name__)


@dataclass
class DeepFact:
    """Факт, извлечённый из глубоких слоёв памяти."""

    memory_id: int
    fact: str
    tier: int
    confidence: float
    reason: str  # "semantic", "graph_bfs", "cluster_match", "distilled"
    contact_id: Optional[int] = None
    tags: list = field(default_factory=list)


@dataclass
class MemoryGraphNode:
    """Узел графа памяти с соседями."""

    memory_id: int
    fact: str
    tier: int
    neighbors: list  # list of (related_memory_id, relation_type)


@dataclass
class DeepRecallResult:
    """Результат глубокой выборки."""

    facts: list[DeepFact]
    graph: list[MemoryGraphNode]  # BFS-expanded nodes with neighbors
    total_explored: int


class DeepMemoryRetrieval:
    """Глубокая выборка из tier-2/3 + BFS по MemoryLink."""

    def __init__(
        self,
        tier2_limit: int = 5,
        tier3_limit: int = 3,
        distilled_limit: int = 2,
        bfs_max_depth: int = 2,
        bfs_max_nodes: int = 15,
        bfs_max_branch: int = 5,
    ):
        self.tier2_limit = tier2_limit
        self.tier3_limit = tier3_limit
        self.distilled_limit = distilled_limit
        self.bfs_max_depth = bfs_max_depth
        self.bfs_max_nodes = bfs_max_nodes
        self.bfs_max_branch = bfs_max_branch

    async def prefetch_facts(
        self,
        session: AsyncSession,
        owner_id: int,
        context_keywords: list[str] | None = None,
        contact_id: int | None = None,
        telegram_id: int | None = None,
    ) -> list[DeepFact]:
        """
        Шаг 1: Префетч фактов из глубоких слоёв памяти.

        Приоритет:
        1. tier-2 (memory_tier=2): фильтр по ключевым словам + use_count
        2. tier-3 (memory_tier=3): cluster_first_retrieval() + прямой запрос
        3. distilled (💡 в tags): отдельный запрос top-N
        """
        facts: list[DeepFact] = []

        # --- Tier 2: medium-term memory (memory_tier=2) ---
        q2 = select(Memory).where(
            Memory.is_active == True,
            Memory.memory_tier == 2,
            Memory.user_id == owner_id,
        )
        if contact_id:
            q2 = q2.where(Memory.contact_id == contact_id)
        if context_keywords:
            conditions = []
            for kw in context_keywords:
                conditions.append(Memory.fact.ilike(f"%{kw}%"))
            if conditions:
                q2 = q2.where(or_(*conditions))
        q2 = q2.order_by(Memory.use_count.desc()).limit(self.tier2_limit)

        result2 = (await session.execute(q2)).scalars().all()
        for m in result2:
            facts.append(
                DeepFact(
                    memory_id=m.id,
                    fact=m.fact,
                    tier=2,
                    confidence=0.7,
                    reason="tier2_prefetch",
                    contact_id=m.contact_id,
                    tags=_parse_tags(m.tags),
                )
            )

        # --- Tier 3: long-term memory (memory_tier=3) ---
        # Используем cluster_first_retrieval если есть ключевые слова
        if context_keywords and telegram_id:
            try:
                from src.core.memory.memory_clusterer import cluster_first_retrieval

                cluster_facts = await cluster_first_retrieval(
                    telegram_id=telegram_id,
                    query=" ".join(context_keywords),
                    limit=self.tier3_limit,
                )
                for cf in cluster_facts[: self.tier3_limit]:
                    if cf.id not in {f.memory_id for f in facts}:
                        facts.append(
                            DeepFact(
                                memory_id=cf.id,
                                fact=cf.fact,
                                tier=3,
                                confidence=0.6,
                                reason="cluster_match",
                                contact_id=cf.contact_id,
                                tags=_parse_tags(cf.tags),
                            )
                        )
            except Exception:
                logger.debug(
                    "cluster_first_retrieval failed, falling back", exc_info=True
                )

        # Fallback: прямой запрос tier-3
        tier3_count = len([f for f in facts if f.tier == 3])
        if tier3_count < self.tier3_limit:
            remaining = self.tier3_limit - tier3_count
            q3 = select(Memory).where(
                Memory.is_active == True,
                Memory.memory_tier == 3,
                Memory.user_id == owner_id,
            )
            if contact_id:
                q3 = q3.where(Memory.contact_id == contact_id)
            q3 = q3.order_by(Memory.use_count.desc()).limit(remaining)
            result3 = (await session.execute(q3)).scalars().all()
            for m in result3:
                if m.id not in {f.memory_id for f in facts}:
                    facts.append(
                        DeepFact(
                            memory_id=m.id,
                            fact=m.fact,
                            tier=3,
                            confidence=0.5,
                            reason="tier3_fallback",
                            contact_id=m.contact_id,
                            tags=_parse_tags(m.tags),
                        )
                    )

        # --- Distilled knowledge (💡 в tags) ---
        q_distilled = (
            select(Memory)
            .where(
                Memory.is_active == True,
                Memory.user_id == owner_id,
                Memory.tags.ilike("%💡%"),
            )
            .order_by(Memory.use_count.desc())
            .limit(self.distilled_limit)
        )
        result_d = (await session.execute(q_distilled)).scalars().all()
        for m in result_d:
            if m.id not in {f.memory_id for f in facts}:
                facts.append(
                    DeepFact(
                        memory_id=m.id,
                        fact=f"💡 {m.fact}",
                        tier=m.memory_tier or 3,
                        confidence=0.9,
                        reason="distilled",
                        contact_id=m.contact_id,
                        tags=_parse_tags(m.tags),
                    )
                )

        # --- L2 Scene narratives (LLM-generated cluster summaries) ---
        try:
            from sqlalchemy import func
            from src.db.models import MemoryCluster

            scene_result = await session.execute(
                select(MemoryCluster.summary, MemoryCluster.topic)
                .where(
                    MemoryCluster.user_id == owner_id,
                    MemoryCluster.summary.isnot(None),
                    MemoryCluster.summary != "",
                    func.length(MemoryCluster.summary) > 20,
                )
                .order_by(MemoryCluster.updated_at.desc())
                .limit(3)
            )
            scene_rows = scene_result.all()
            for summary, topic in scene_rows:
                if summary:
                    facts.append(
                        DeepFact(
                            memory_id=0,
                            fact=f"[Сцена: {topic}] {summary}",
                            tier=2,
                            confidence=0.75,
                            reason="scene_narrative",
                        )
                    )
        except Exception:
            logger.debug("Scene narratives not available", exc_info=True)

        return facts

    async def bfs_expand(
        self,
        session: AsyncSession,
        seed_memory_ids: list[int],
        owner_id: int,
    ) -> list[MemoryGraphNode]:
        """
        Шаг 2: BFS по графу MemoryLink от seed-фактов.

        Использует MemoryLink.source_id → target_id для обхода.
        """
        if not seed_memory_ids:
            return []

        visited: set[int] = set(seed_memory_ids)
        nodes: dict[int, MemoryGraphNode] = {}
        frontier = list(seed_memory_ids)
        total_nodes = len(seed_memory_ids)

        # Загружаем seed факты
        q_seed = select(Memory).where(
            Memory.id.in_(seed_memory_ids),
            Memory.user_id == owner_id,
        )
        seed_facts = (await session.execute(q_seed)).scalars().all()
        for m in seed_facts:
            nodes[m.id] = MemoryGraphNode(
                memory_id=m.id,
                fact=m.fact,
                tier=m.memory_tier or 1,
                neighbors=[],
            )

        for _depth in range(self.bfs_max_depth):
            if total_nodes >= self.bfs_max_nodes:
                break

            next_frontier: list[int] = []
            for node_id in frontier:
                if total_nodes >= self.bfs_max_nodes:
                    break

                # Найти все связи ОТ этого узла (source_id = node_id)
                q_links = (
                    select(MemoryLink)
                    .where(
                        MemoryLink.source_id == node_id,
                        MemoryLink.user_id == owner_id,
                    )
                    .limit(self.bfs_max_branch)
                )
                links = (await session.execute(q_links)).scalars().all()

                for link in links:
                    neighbor_id = link.target_id
                    if neighbor_id not in visited and total_nodes < self.bfs_max_nodes:
                        visited.add(neighbor_id)
                        next_frontier.append(neighbor_id)
                        total_nodes += 1

                        if node_id in nodes:
                            nodes[node_id].neighbors.append(
                                (neighbor_id, link.relation_type or "related")
                            )

                    elif neighbor_id in nodes and node_id in nodes:
                        # Уже посещён — всё равно записываем связь если нет
                        existing = [n[0] for n in nodes[node_id].neighbors]
                        if neighbor_id not in existing:
                            nodes[node_id].neighbors.append(
                                (neighbor_id, link.relation_type or "related")
                            )

            if next_frontier:
                # Загружаем факты новых узлов
                q_new = select(Memory).where(
                    Memory.id.in_(next_frontier),
                    Memory.user_id == owner_id,
                )
                new_facts = (await session.execute(q_new)).scalars().all()
                for m in new_facts:
                    if m.id not in nodes:
                        nodes[m.id] = MemoryGraphNode(
                            memory_id=m.id,
                            fact=m.fact,
                            tier=m.memory_tier or 1,
                            neighbors=[],
                        )

            frontier = next_frontier

        return list(nodes.values())

    def format_deep_context(
        self,
        facts: list[DeepFact],
        graph: list[MemoryGraphNode],
    ) -> str:
        """
        Форматирует результат глубокой выборки для вставки в system prompt.
        """
        parts: list[str] = ["<deep_memory>"]

        if facts:
            tier2 = [f for f in facts if f.tier == 2]
            tier3 = [f for f in facts if f.tier == 3]
            distilled = [f for f in facts if f.reason == "distilled"]

            if tier2:
                parts.append("<tier2_facts>")
                for f in tier2:
                    parts.append(
                        f"- [{f.reason}] {f.fact} (confidence={f.confidence:.1f})"
                    )
                parts.append("</tier2_facts>")

            if tier3:
                parts.append("<tier3_facts>")
                for f in tier3:
                    parts.append(
                        f"- [{f.reason}] {f.fact} (confidence={f.confidence:.1f})"
                    )
                parts.append("</tier3_facts>")

            if distilled:
                parts.append("<distilled_knowledge>")
                for f in distilled:
                    parts.append(f"- {f.fact}")
                parts.append("</distilled_knowledge>")

            # Scene narratives
            scene_facts = [f for f in facts if f.reason == "scene_narrative"]
            if scene_facts:
                parts.append("<scene_narratives>")
                for f in scene_facts:
                    parts.append(f"  <scene>{f.fact}</scene>")
                parts.append("</scene_narratives>")

        if graph:
            parts.append("<memory_links>")
            for node in graph:
                for neighbor_id, rel_type in node.neighbors:
                    neighbor_node = next(
                        (n for n in graph if n.memory_id == neighbor_id), None
                    )
                    if neighbor_node:
                        parts.append(
                            f"- [{rel_type}] {node.fact[:80]}... → "
                            f"{neighbor_node.fact[:80]}..."
                        )
            parts.append("</memory_links>")

        parts.append("</deep_memory>")
        return "\n".join(parts)

    async def retrieve(
        self,
        session: AsyncSession,
        owner_id: int,
        context_keywords: list[str] | None = None,
        contact_id: int | None = None,
        telegram_id: int | None = None,
    ) -> DeepRecallResult:
        """
        Полный пайплайн: префетч + BFS.
        """
        facts = await self.prefetch_facts(
            session=session,
            owner_id=owner_id,
            context_keywords=context_keywords,
            contact_id=contact_id,
            telegram_id=telegram_id,
        )
        seed_ids = [f.memory_id for f in facts]
        graph = await self.bfs_expand(
            session=session,
            seed_memory_ids=seed_ids,
            owner_id=owner_id,
        )
        return DeepRecallResult(
            facts=facts,
            graph=graph,
            total_explored=len(graph),
        )


def _parse_tags(tags_str: str | None) -> list:
    """Парсит comma-separated строку тегов в список."""
    if not tags_str:
        return []
    return [t.strip() for t in tags_str.split(",") if t.strip()]


def _extract_keywords(text: str, max_keywords: int = 5) -> list[str]:
    """
    Извлекает ключевые слова из текста пользователя.

    Простая реализация: разбивает на слова, фильтрует короткие/стоп-слова,
    возвращает самые длинные значимые слова.
    """
    if not text:
        return []

    STOP_WORDS = {
        "я",
        "ты",
        "он",
        "она",
        "оно",
        "мы",
        "вы",
        "они",
        "мой",
        "твой",
        "его",
        "её",
        "наш",
        "ваш",
        "их",
        "это",
        "этот",
        "эта",
        "эти",
        "тот",
        "та",
        "те",
        "что",
        "как",
        "где",
        "когда",
        "почему",
        "зачем",
        "кто",
        "не",
        "ни",
        "бы",
        "же",
        "ли",
        "и",
        "а",
        "но",
        "или",
        "в",
        "на",
        "с",
        "по",
        "к",
        "из",
        "от",
        "до",
        "для",
        "у",
        "за",
        "над",
        "под",
        "при",
        "про",
        "без",
        "через",
        "был",
        "была",
        "было",
        "были",
        "есть",
        "будет",
        "будут",
        "может",
        "могут",
        "надо",
        "нужно",
        "можно",
        "хочу",
        "да",
        "нет",
        "всё",
        "весь",
        "вся",
        "все",
    }

    # Разбиваем на слова, оставляем только буквенные длиной > 2
    words = text.lower().split()
    keywords = []
    for w in words:
        # Убираем знаки препинания
        clean = "".join(c for c in w if c.isalpha())
        if len(clean) > 2 and clean not in STOP_WORDS:
            keywords.append(clean)

    # Дубликаты убираем, сортируем по длине (более длинные — более значимые)
    seen: set[str] = set()
    unique = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)

    unique.sort(key=len, reverse=True)
    return unique[:max_keywords]


# Синглтон
deep_memory = DeepMemoryRetrieval()
