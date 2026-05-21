"""
MemoryRecallService — единый «мозг» памяти.
Объединяет: contact facts + self facts + Qdrant semantic + pinned + fresh + task-context.
Возвращает факты с причиной включения.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.db.repo import get_or_create_user, get_contact
from src.db.session import get_session
from src.llm.router import build_provider
from src.core.memory.hybrid_search import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

UTC_NAIVE = lambda: datetime.now(timezone.utc).replace(tzinfo=None)


@dataclass
class RecalledFact:
    """Факт, извлечённый recall-сервисом, с причиной попадания."""

    fact: str
    reason: str  # "pinned / similar to query / свежий / часто использовался / task-context / self / contact"
    confidence: float = 0.5
    memory_id: int | None = None
    contact_id: int | None = None
    layer: str = "recent"


@dataclass
class RecallResult:
    facts: list[RecalledFact] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


async def recall(
    telegram_id: int,
    *,
    session=None,  # NEW: опциональная сессия извне
    contact_id: int | None = None,
    query: str | None = None,
    limit: int = 8,
    include_self: bool = True,
    include_pinned: bool = True,
    include_tasks: bool = True,
    include_deep: bool = True,
    semantic_threshold: float = 0.55,
    mode: str = "deep",
) -> RecallResult:
    """
    Единый recall-сервис памяти.

    Приоритет:
    1. pinned-факты (всегда первые)
    2. task-context (привязанные к активным обязательствам)
    3. Qdrant-semantic (похожие на query)
    4. fresh — свежие за 7 дней с высокой уверенностью
    5. frequently-used — высокий use_count
    6. self-факты (глобальные, без contact_id)
    7. contact-факты (связанные с конкретным контактом)
    8. deep — tier 2-3 префетч + BFS по MemoryLink графу (опционально)

    Возвращает список RecalledFact с причинами.
    """
    result = RecallResult()
    mode = (mode or "deep").lower()
    if mode not in {"light", "normal", "deep"}:
        mode = "deep"
    include_deep = include_deep and mode == "deep"
    include_semantic = bool(query) and mode in {"normal", "deep"}
    include_frequent = mode in {"normal", "deep"}
    include_self_facts = include_self and mode in {"normal", "deep"}
    include_contact_facts = mode in {"normal", "deep"}
    now = UTC_NAIVE()
    seen_ids: set[int] = set()
    ranked: list[RecalledFact] = []

    _close_session = session is None
    _session_cm = None
    if session is None:
        _session_cm = get_session()
        session = await _session_cm.__aenter__()
    try:
        owner = await get_or_create_user(session, telegram_id)
        from src.db.models import Memory
        from sqlalchemy import select, or_

        # Все активные факты пользователя
        base_conditions = [
            Memory.user_id == owner.id,
            Memory.is_active == True,
            or_(Memory.expires_at.is_(None), Memory.expires_at > now),
        ]
        q_all = (
            select(Memory)
            .where(*base_conditions)
            .order_by(
                Memory.pinned.desc(),
                Memory.created_at.desc(),
                Memory.confidence.desc(),
            )
        )
        if mode == "light":
            q_all = q_all.limit(max(limit * 8, 40))
        elif mode == "normal":
            q_all = q_all.limit(max(limit * 20, 160))
        all_facts_result = await session.execute(q_all)
        all_facts: list[Memory] = list(all_facts_result.scalars().all())

        # --- 1. Pinned ---
        if include_pinned:
            pinned = [m for m in all_facts if m.pinned and m.id not in seen_ids]
            for m in pinned:
                ranked.append(
                    RecalledFact(
                        fact=m.fact,
                        reason="📌 закреплён",
                        confidence=m.confidence or 0.5,
                        memory_id=m.id,
                        contact_id=m.contact_id,
                        layer=m.temporal_layer or "recent",
                    )
                )
                seen_ids.add(m.id)

        # --- 2. Task-context ---
        if include_tasks:
            from src.db.models import Commitment

            task_facts = [
                m for m in all_facts if m.memory_type == "task" and m.id not in seen_ids
            ]
            if task_facts:
                # Проверяем, есть ли активный commitment для этих фактов
                task_ids = [m.id for m in task_facts]
                commits_result = await session.execute(
                    select(Commitment).where(
                        Commitment.source_memory_id.in_(task_ids),
                        Commitment.status == "open",
                    )
                )
                active_task_ids = {
                    c.source_memory_id for c in commits_result.scalars().all()
                }
                for m in task_facts:
                    if m.id in active_task_ids:
                        ranked.append(
                            RecalledFact(
                                fact=m.fact,
                                reason="📋 активная задача",
                                confidence=m.confidence or 0.5,
                                memory_id=m.id,
                                contact_id=m.contact_id,
                                layer=m.temporal_layer or "recent",
                            )
                        )
                        seen_ids.add(m.id)

        # --- 3. Hybrid search: Qdrant semantic + FTS5 keyword (RRF) ---
        if include_semantic:
            try:
                from src.core.actions.vector_store import get_vector_store
                from src.db.repo import search_memories_fts_with_scores

                provider = await build_provider(session, owner)
                if provider:
                    embedding = await provider.embed(query[:300])

                    # Параллельный запуск: векторный + ключевой поиск
                    vector_task = get_vector_store().search_similar_memories(
                        user_id=owner.id,
                        embedding=embedding,
                        threshold=semantic_threshold,
                        limit=10,
                        contact_id=contact_id,
                    )
                    keyword_task = search_memories_fts_with_scores(
                        session,
                        owner,
                        query,
                        contact_id=contact_id,
                        limit=10,
                    )

                    vector_hits_raw, keyword_hits_raw = await asyncio.gather(
                        vector_task,
                        keyword_task,
                    )

                    # Преобразуем в (memory_id, score) для RRF
                    vector_hits: list[tuple[int, float]] = [
                        (h["memory_id"], h["score"])
                        for h in vector_hits_raw
                        if h.get("memory_id") is not None
                    ]
                    keyword_hits: list[tuple[int, float]] = keyword_hits_raw

                    # Reciprocal Rank Fusion
                    fused = reciprocal_rank_fusion(
                        vector_results=vector_hits,
                        keyword_results=keyword_hits,
                    )

                    for mem_id, fused_score in fused:
                        if mem_id not in seen_ids:
                            m = next((f for f in all_facts if f.id == mem_id), None)
                            if m:
                                ranked.append(
                                    RecalledFact(
                                        fact=m.fact,
                                        reason="🔍 гибридный поиск",
                                        confidence=round(fused_score, 3),
                                        memory_id=m.id,
                                        contact_id=m.contact_id,
                                        layer=m.temporal_layer or "recent",
                                    )
                                )
                                seen_ids.add(m.id)
            except (ImportError, ValueError, ConnectionError, OSError):
                logger.debug("Hybrid recall failed, skipping", exc_info=True)

        # --- 4. Fresh (7 days, high confidence) ---
        cutoff_7d = UTC_NAIVE()
        from datetime import timedelta

        cutoff_7d = cutoff_7d - timedelta(days=7)
        fresh = [
            m
            for m in all_facts
            if m.id not in seen_ids
            and m.created_at
            and m.created_at >= cutoff_7d
            and (m.confidence or 0) >= 0.5
        ]
        fresh.sort(key=lambda m: m.confidence or 0, reverse=True)
        for m in fresh[:3]:
            ranked.append(
                RecalledFact(
                    fact=m.fact,
                    reason="🆕 свежий",
                    confidence=m.confidence or 0.5,
                    memory_id=m.id,
                    contact_id=m.contact_id,
                    layer=m.temporal_layer or "recent",
                )
            )
            seen_ids.add(m.id)

        # --- 5. Frequently used ---
        freq = [
            m for m in all_facts if m.id not in seen_ids and (m.use_count or 0) >= 3
        ]
        freq.sort(key=lambda m: m.use_count or 0, reverse=True)
        for m in freq[:2]:
            ranked.append(
                RecalledFact(
                    fact=m.fact,
                    reason=f"⭐ часто (×{m.use_count})",
                    confidence=m.confidence or 0.5,
                    memory_id=m.id,
                    contact_id=m.contact_id,
                    layer=m.temporal_layer or "recent",
                )
            )
            seen_ids.add(m.id)

        # --- 6. Self-facts (глобальные) ---
        if include_self_facts:
            self_facts = [
                m for m in all_facts if m.id not in seen_ids and m.contact_id is None
            ]
            self_facts.sort(key=lambda m: m.confidence or 0, reverse=True)
            for m in self_facts[:2]:
                ranked.append(
                    RecalledFact(
                        fact=m.fact,
                        reason="🧑 о тебе",
                        confidence=m.confidence or 0.5,
                        memory_id=m.id,
                        contact_id=m.contact_id,
                        layer=m.temporal_layer or "recent",
                    )
                )
                seen_ids.add(m.id)

        # --- 7. Contact-specific facts ---
        if contact_id and include_contact_facts:
            contact_facts = [
                m
                for m in all_facts
                if m.id not in seen_ids and m.contact_id == contact_id
            ]
            contact_facts.sort(key=lambda m: m.confidence or 0, reverse=True)
            for m in contact_facts[:5]:
                ranked.append(
                    RecalledFact(
                        fact=m.fact,
                        reason="👤 о контакте",
                        confidence=m.confidence or 0.5,
                        memory_id=m.id,
                        contact_id=m.contact_id,
                        layer=m.temporal_layer or "recent",
                    )
                )
                seen_ids.add(m.id)

        # --- 8. Deep memory: tier 2-3 prefetch + BFS graph expansion ---
        if include_deep:
            try:
                from src.core.memory.deep_memory import (
                    deep_memory as dm,
                    _extract_keywords,
                )

                keywords = _extract_keywords(query) if query else None
                deep_result = await dm.retrieve(
                    session=session,
                    owner_id=owner.id,
                    context_keywords=keywords,
                    contact_id=contact_id,
                    telegram_id=telegram_id,
                )
                for f in deep_result.facts:
                    if f.memory_id not in seen_ids:
                        ranked.append(
                            RecalledFact(
                                fact=f.fact,
                                reason=f"🧠 deep:{f.reason}",
                                confidence=f.confidence,
                                memory_id=f.memory_id,
                                contact_id=f.contact_id,
                                layer="deep",
                            )
                        )
                        seen_ids.add(f.memory_id)
                # Сохраняем граф для форматирования
                result.meta["deep_graph"] = deep_result.graph
            except (ImportError, ValueError, ConnectionError, OSError):
                logger.debug("Deep memory recall failed, skipping", exc_info=True)

        # Limit
        result.facts = ranked[:limit]
        result.meta |= {
            "mode": mode,
            "total_active": len(all_facts),
            "returned": len(result.facts),
            "reasons_used": list(set(f.reason for f in result.facts)),
        }

        # Инкрементируем use_count для возвращённых фактов
        for f in result.facts:
            if f.memory_id:
                m = next((x for x in all_facts if x.id == f.memory_id), None)
                if m:
                    m.use_count = (m.use_count or 0) + 1
                    m.last_used_at = now
        await session.flush()
    finally:
        if _close_session and _session_cm is not None:
            await _session_cm.__aexit__(None, None, None)

    return result


def format_recall_for_prompt(recall_result: RecallResult, max_facts: int = 8) -> str:
    """Форматирует результат recall для инжекции в LLM-промпт."""
    if not recall_result.facts:
        return ""

    deep_facts = [rf for rf in recall_result.facts if rf.layer == "deep"]
    surface_facts = [rf for rf in recall_result.facts if rf.layer != "deep"]

    lines: list[str] = []

    # Поверхностная память (шаги 1-7)
    if surface_facts:
        lines.append("<recall_context>")
        for rf in surface_facts[:max_facts]:
            lines.append(f"[{rf.reason}] {rf.fact}")
        lines.append("</recall_context>")

    # Глубокая память (шаг 8)
    if deep_facts:
        lines.append('<recall_context type="deep">')
        for rf in deep_facts[:max_facts]:
            lines.append(f"[{rf.reason}] {rf.fact}")
        lines.append("</recall_context>")

    # Граф MemoryLink (если есть)
    deep_graph = recall_result.meta.get("deep_graph")
    if deep_graph:
        try:
            from src.core.memory.deep_memory import deep_memory as dm

            graph_context = dm.format_deep_context(
                facts=[],  # факты уже отформатированы выше
                graph=deep_graph,
            )
            if graph_context and "<memory_links>" in graph_context:
                # Извлекаем только <memory_links> секцию
                start = graph_context.find("<memory_links>")
                end_marker = graph_context.find("</memory_links>")
                if start != -1 and end_marker != -1:
                    end = end_marker + len("</memory_links>")
                    lines.append(graph_context[start:end])
        except Exception:
            logger.debug(
                "memory_recall: graph_context extraction failed", exc_info=True
            )
            pass

    return "\n".join(lines)


def format_recall_human(recall_result: RecallResult, max_facts: int = 5) -> str:
    """Форматирует результат recall для показа пользователю."""
    if not recall_result.facts:
        return "Память пуста."
    lines = ["🧠 <b>Релевантная память:</b>"]
    for rf in recall_result.facts[:max_facts]:
        lines.append(f"{rf.reason}: {rf.fact[:100]}")
    return "\n".join(lines)
