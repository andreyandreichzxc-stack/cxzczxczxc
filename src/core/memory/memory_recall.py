"""
MemoryRecallService — единый «мозг» памяти.
Объединяет: contact facts + self facts + Qdrant semantic + pinned + fresh + task-context.
Возвращает факты с причиной включения.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
import time

from src.config import settings
from src.db.repo import get_or_create_user
from src.db.session import get_session
from src.llm.router import _ensure_utc, build_provider
from src.core.memory.hybrid_search import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

_recall_cache: dict[str, tuple[float, RecallResult]] = {}
_recall_lock: asyncio.Lock = asyncio.Lock()
_RECALL_CACHE_MAX = settings.max_recall_cache_size
_RECALL_CACHE_RESULT_TTL = 30.0  # TTL for results WITH facts
_RECALL_CACHE_EMPTY_TTL = 60.0  # TTL for empty results
_bg_bump_tasks: list[asyncio.Task] = []  # track background bump tasks


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Jaccard similarity on word sets — fallback when embeddings unavailable."""
    set_a = set(text_a.lower().split())
    set_b = set(text_b.lower().split())
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def _mmr_rerank(
    facts: list[dict],
    query_embedding: list[float] | None = None,
    lambda_param: float = 0.7,
    top_k: int | None = None,
) -> list[dict]:
    """
    Maximal Marginal Relevance re-ranking.

    Balances **relevance** (score) vs **diversity** (dissimilarity between
    selected items) so the bot doesn't present nearly-identical facts.

    Each dict in *facts* must have:
        - "score" (float): original relevance score
        - "fact"  (str):  fact text

    Algorithm
    ---------
    1. Start with the highest-scoring fact.
    2. For each remaining candidate compute:
       MMR = λ · relevance − (1 − λ) · max_similarity_to_already_selected
    3. Greedily pick the best candidate.
    """
    if not facts:
        return facts

    if top_k is None:
        top_k = len(facts)

    sorted_facts = sorted(facts, key=lambda x: x.get("score", 0), reverse=True)
    selected = [sorted_facts[0]]
    candidates = sorted_facts[1:]

    while candidates and len(selected) < top_k:
        best_idx = -1
        best_score = -float("inf")

        for i, cand in enumerate(candidates):
            # Relevance: use pre-computed score (e.g. RRF fused score)
            relevance = cand.get("score", 0)

            # Diversity: max Jaccard similarity to any already-selected fact
            max_sim = 0.0
            for sel in selected:
                sim = _jaccard_similarity(cand["fact"], sel["fact"])
                if sim > max_sim:
                    max_sim = sim

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        if best_idx >= 0:
            selected.append(candidates.pop(best_idx))
        else:
            break

    return selected


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


async def _bump_use_count_async(
    telegram_id: int,
    result: RecallResult,
) -> None:
    """Fire-and-forget: increment use_count for cached results."""
    try:
        async with get_session() as session:
            from src.db.models import Memory

            now_cache = datetime.now(timezone.utc)
            for f in result.facts:
                if f.memory_id:
                    m = await session.get(Memory, f.memory_id)
                    if m:
                        m.use_count = (m.use_count or 0) + 1
                        m.last_used_at = now_cache
            await session.flush()
    except Exception:
        logger.debug("Async use_count bump failed", exc_info=True)


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
    global _bg_bump_tasks
    _cache_key = f"{telegram_id}:{query}:{contact_id}:{mode}"
    _cache_now = time.monotonic()
    async with _recall_lock:
        if _cache_key in _recall_cache:
            ts, cached = _recall_cache[_cache_key]
            # Check TTL based on whether result has facts
            ttl = _RECALL_CACHE_RESULT_TTL if cached.facts else _RECALL_CACHE_EMPTY_TTL
            if _cache_now - ts < ttl:
                # Async increment use_count — don't block the return
                if cached.facts:
                    _bg_bump_tasks.append(
                        asyncio.create_task(_bump_use_count_async(telegram_id, cached))
                    )
                    # Clean done tasks periodically
                    if len(_bg_bump_tasks) > 20:
                        _bg_bump_tasks = [t for t in _bg_bump_tasks if not t.done()]
                return cached
            else:
                del _recall_cache[_cache_key]

    result = RecallResult()
    mode = (mode or "deep").lower()
    if mode not in {"light", "normal", "deep"}:
        mode = "deep"
    include_deep = include_deep and mode == "deep"
    include_semantic = bool(query) and mode in {"normal", "deep"}
    _include_frequent = mode in {"normal", "deep"}
    include_self_facts = include_self and mode in {"normal", "deep"}
    include_contact_facts = mode in {"normal", "deep"}
    now = _utc_now()
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
            Memory.is_active,
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
        else:  # deep
            q_all = q_all.limit(max(limit * 40, 500))
        all_facts_result = await session.execute(q_all)
        all_facts: list[Memory] = list(all_facts_result.scalars().all())

        # --- 1. Pinned ---
        if include_pinned:
            pinned = [
                m
                for m in all_facts
                if m.pinned
                and m.id not in seen_ids
                and (
                    not contact_id or m.contact_id is None or m.contact_id == contact_id
                )
            ]
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

                    # Лимит поиска зависит от режима: light → 0 (не вызывается),
                    # normal → 5, deep → 10
                    qdrant_limit = {
                        "light": 0,
                        "normal": 5,
                        "deep": 10,
                    }.get(mode, 10)

                    # Параллельный запуск: векторный + ключевой поиск
                    vector_task = get_vector_store().search_similar_memories(
                        user_id=owner.id,
                        embedding=embedding,
                        threshold=semantic_threshold,
                        limit=qdrant_limit,
                        contact_id=contact_id,
                    )
                    keyword_task = search_memories_fts_with_scores(
                        session,
                        owner,
                        query,
                        contact_id=contact_id,
                        limit=qdrant_limit,
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

                    hybrid_ranked: list[RecalledFact] = []
                    for mem_id, fused_score in fused:
                        if mem_id not in seen_ids:
                            m = next((f for f in all_facts if f.id == mem_id), None)
                            if m:
                                hybrid_ranked.append(
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

                    # MMR rerank: balance relevance vs diversity
                    if len(hybrid_ranked) > 2:
                        mmr_input = [
                            {"score": rf.confidence, "fact": rf.fact}
                            for rf in hybrid_ranked
                        ]
                        mmr_output = _mmr_rerank(
                            mmr_input,
                            query_embedding=embedding,
                        )
                        # Reorder hybrid_ranked to match MMR ranking
                        mmr_rank = {d["fact"]: idx for idx, d in enumerate(mmr_output)}
                        hybrid_ranked.sort(
                            key=lambda rf: mmr_rank.get(rf.fact, float("inf"))
                        )

                    ranked.extend(hybrid_ranked)
            except (ImportError, ValueError, ConnectionError, OSError):
                logger.debug("Hybrid recall failed, skipping", exc_info=True)

        # --- 4. Fresh (7 days, high confidence) ---
        cutoff_7d = _utc_now()
        from datetime import timedelta

        cutoff_7d = cutoff_7d - timedelta(days=7)
        fresh = [
            m
            for m in all_facts
            if m.id not in seen_ids
            and (ca := _ensure_utc(m.created_at))
            and ca >= cutoff_7d
            and (m.confidence or 0) >= 0.5
            and (m.contact_id == contact_id if contact_id else True)
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
            m
            for m in all_facts
            if m.id not in seen_ids
            and (m.use_count or 0) >= 3
            and (m.contact_id == contact_id if contact_id else True)
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

        # Cache ALL results (before use_count increment so cache is clean).
        # On cache hit, _bump_use_count_async fires a background increment.
        async with _recall_lock:
            if len(_recall_cache) >= _RECALL_CACHE_MAX:
                # Evict 10% of oldest entries (not just 1)
                evict_count = max(1, int(_RECALL_CACHE_MAX * 0.1))
                sorted_items = sorted(_recall_cache.items(), key=lambda x: x[1][0])
                for i in range(evict_count):
                    if i < len(sorted_items):
                        del _recall_cache[sorted_items[i][0]]
            _recall_cache[_cache_key] = (_cache_now, result)

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
            await _session_cm.__aexit__(*sys.exc_info())

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
