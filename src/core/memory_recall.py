"""
MemoryRecallService — единый «мозг» памяти.
Объединяет: contact facts + self facts + Qdrant semantic + pinned + fresh + task-context.
Возвращает факты с причиной включения.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.db.session import get_session
from src.db.repo import get_or_create_user, get_contact

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
    contact_id: int | None = None,
    query: str | None = None,
    limit: int = 8,
    include_self: bool = True,
    include_pinned: bool = True,
    include_tasks: bool = True,
    semantic_threshold: float = 0.55,
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

    Возвращает список RecalledFact с причинами.
    """
    result = RecallResult()
    now = UTC_NAIVE()
    seen_ids: set[int] = set()
    ranked: list[RecalledFact] = []

    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)
        from src.db.models import Memory
        from sqlalchemy import select, or_

        # Все активные факты пользователя
        base_conditions = [
            Memory.user_id == owner.id,
            Memory.is_active == True,
            or_(Memory.expires_at.is_(None), Memory.expires_at > now),
        ]
        all_facts_result = await session.execute(
            select(Memory).where(*base_conditions).order_by(Memory.confidence.desc())
        )
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

        # --- 3. Qdrant semantic ---
        if query:
            try:
                from src.core.vector_store import vector_store
                from src.llm.router import build_provider

                provider = await build_provider(session, owner)
                if provider:
                    embedding = await provider.embed(query[:300])
                    semantic_hits = await vector_store.search_similar_memories(
                        user_id=owner.id,
                        embedding=embedding,
                        threshold=semantic_threshold,
                        limit=5,
                        contact_id=contact_id,
                    )
                    for hit in semantic_hits:
                        mid = hit.get("memory_id")
                        if mid and mid not in seen_ids:
                            m = next((f for f in all_facts if f.id == mid), None)
                            if m:
                                ranked.append(
                                    RecalledFact(
                                        fact=m.fact,
                                        reason="🔍 похож на запрос",
                                        confidence=m.confidence or 0.5,
                                        memory_id=m.id,
                                        contact_id=m.contact_id,
                                        layer=m.temporal_layer or "recent",
                                    )
                                )
                                seen_ids.add(m.id)
            except Exception:
                logger.debug("Semantic recall failed, skipping", exc_info=True)

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
        if include_self:
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
        if contact_id:
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

        # Limit
        result.facts = ranked[:limit]
        result.meta = {
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

    return result


def format_recall_for_prompt(recall_result: RecallResult, max_facts: int = 8) -> str:
    """Форматирует результат recall для инжекции в LLM-промпт."""
    if not recall_result.facts:
        return ""
    lines = ["<recall_context>"]
    for rf in recall_result.facts[:max_facts]:
        lines.append(f"[{rf.reason}] {rf.fact}")
    lines.append("</recall_context>")
    return "\n".join(lines)


def format_recall_human(recall_result: RecallResult, max_facts: int = 5) -> str:
    """Форматирует результат recall для показа пользователю."""
    if not recall_result.facts:
        return "Память пуста."
    lines = ["🧠 <b>Релевантная память:</b>"]
    for rf in recall_result.facts[:max_facts]:
        lines.append(f"{rf.reason}: {rf.fact[:100]}")
    return "\n".join(lines)
