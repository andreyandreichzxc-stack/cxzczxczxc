"""Memory repository — Memory, MemoryLink, MemoryCluster, MemoryCandidate, FTS."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import case, delete, distinct, func, or_, select, text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import (
    Contact,
    Memory,
    MemoryCandidate,
    MemoryCluster,
    MemoryClusterMember,
    MemoryLink,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.actions.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class FtsHit:
    user_id: int
    peer_id: int
    message_id: int
    sender_name: str | None
    snippet: str
    rank: float
    peer_name: str | None = None
    date: datetime | None = None


def _fts_query_for(query: str) -> str:
    """Build an FTS5-safe MATCH expression from free-text user query.

    Each word becomes a prefix-match joined with OR.
    FTS5 operator keywords (OR, AND, NOT, NEAR) are double-quoted to
    prevent them from being interpreted as query operators — this is the
    standard SQLite FTS5 escaping mechanism for literal keyword search.
    """
    _FTS5_KEYWORDS = frozenset({"or", "and", "not", "near"})

    parts: list[str] = []
    for raw in query.split():
        clean = "".join(ch for ch in raw if ch.isalnum() or ch in "_-")
        if len(clean) < 2:
            continue
        lower = clean.lower()
        if lower in _FTS5_KEYWORDS:
            parts.append(f'"{lower}"')
        else:
            parts.append(lower + "*")
    if not parts:
        return ""
    return " OR ".join(parts)


async def fts_search(
    session: AsyncSession,
    user_id: int,
    query: str,
    *,
    limit: int = 50,
) -> list[FtsHit]:
    fts_q = _fts_query_for(query)
    if not fts_q:
        return []
    sql = """
        SELECT m.user_id, m.peer_id, m.message_id, m.sender_name,
               snippet(messages_fts, -1, '', '', '…', 16) AS snippet,
               bm25(messages_fts) AS rank,
               c.display_name AS peer_name,
               m.date
        FROM messages_fts
        JOIN messages m ON m.id = messages_fts.rowid
        LEFT JOIN contacts c ON c.user_id = m.user_id AND c.peer_id = m.peer_id
        WHERE messages_fts MATCH :q AND m.user_id = :uid
        ORDER BY rank
        LIMIT :lim
    """
    result = await session.execute(
        sql_text(sql),
        {"q": fts_q, "uid": user_id, "lim": limit},
    )
    rows = result.mappings().all()
    return [
        FtsHit(
            user_id=int(r["user_id"]),
            peer_id=int(r["peer_id"]),
            message_id=int(r["message_id"]),
            sender_name=r["sender_name"],
            snippet=r["snippet"] or "",
            rank=float(r["rank"]) if r["rank"] is not None else 0.0,
            peer_name=r["peer_name"],
            date=r["date"],
        )
        for r in rows
    ]


async def cross_chat_search(
    session: AsyncSession,
    user,
    query: str,
    limit: int = 5,
    *,
    peer_id: int | None = None,
) -> list[dict]:
    """Cross-chat FTS5 search — searches ALL messages and returns top conversations.

    For each matching conversation returns:
      - peer_id, display_name (from Contact)
      - top 2-3 snippets with highlighted matches (via FTS5 snippet())
      - total matching messages count

    Results are ordered by total matches DESC.

    Args:
        session: DB session.
        user: Bot user.
        query: Free-text search query (each word becomes a prefix OR-match).
        limit: Max number of conversations to return.
        peer_id: Optional — scope search to a single peer/chat.

    Returns:
        List of dicts with keys:
          peer_id, display_name, total_matches, snippets
        Each snippet is a dict: {"sender_name": str | None, "text": str, "date": datetime | None}
    """
    fts_q = _fts_query_for(query)
    if not fts_q:
        return []

    # ── Step 1: find top peer_ids by match count ──────────────────────
    peer_filter = " AND m.peer_id = :pid" if peer_id is not None else ""
    count_sql = f"""
        SELECT m.peer_id, c.display_name, COUNT(*) AS total_matches
        FROM messages_fts
        JOIN messages m ON m.id = messages_fts.rowid
        LEFT JOIN contacts c ON c.user_id = m.user_id AND c.peer_id = m.peer_id
        WHERE messages_fts MATCH :q AND m.user_id = :uid{peer_filter}
        GROUP BY m.peer_id
        ORDER BY total_matches DESC
        LIMIT :lim
    """
    count_params: dict[str, object] = {"q": fts_q, "uid": user.id, "lim": limit}
    if peer_id is not None:
        count_params["pid"] = peer_id

    result = await session.execute(sql_text(count_sql), count_params)
    rows = result.mappings().all()
    if not rows:
        return []

    peer_ids: list[int] = []
    peer_info: dict[int, tuple[str | None, int]] = {}
    for r in rows:
        pid = int(r["peer_id"])
        peer_ids.append(pid)
        peer_info[pid] = (r["display_name"], int(r["total_matches"]))

    # ── Step 2: fetch top-3 snippets per peer_id ────────────────────
    # Build a dynamic IN clause for the selected peer_ids
    placeholders = ", ".join(f":pid_{i}" for i in range(len(peer_ids)))
    params: dict[str, object] = {"q": fts_q, "uid": user.id}
    for i, pid in enumerate(peer_ids):
        params[f"pid_{i}"] = pid

    peer_filter_snippet = " AND m.peer_id = :pid" if peer_id is not None else ""
    snippet_sql = f"""
        SELECT m.peer_id, m.sender_name, m.date,
               snippet(messages_fts, -1, '<b>', '</b>', '…', 64) AS snippet
        FROM messages_fts
        JOIN messages m ON m.id = messages_fts.rowid
        WHERE messages_fts MATCH :q AND m.user_id = :uid{peer_filter_snippet}
          AND m.peer_id IN ({placeholders})
        ORDER BY m.peer_id, bm25(messages_fts)
    """
    if peer_id is not None:
        params["pid"] = peer_id

    result = await session.execute(sql_text(snippet_sql), params)
    snippet_rows = result.mappings().all()

    snippets_by_peer: dict[int, list[dict]] = {}
    for r in snippet_rows:
        pid = int(r["peer_id"])
        if pid not in snippets_by_peer:
            snippets_by_peer[pid] = []
        if len(snippets_by_peer[pid]) < 3:
            snippets_by_peer[pid].append(
                {
                    "sender_name": r["sender_name"],
                    "text": r["snippet"] or "",
                    "date": r["date"],
                }
            )

    # ── Step 3: build result preserving peer order ──────────────────
    output: list[dict] = []
    for pid in peer_ids:
        display_name, total = peer_info[pid]
        output.append(
            {
                "peer_id": pid,
                "display_name": display_name,
                "total_matches": total,
                "snippets": snippets_by_peer.get(pid, []),
            }
        )

    return output


async def add_memory(
    session: AsyncSession,
    user,
    *,
    fact: str,
    contact_id: int | None = None,
    sentiment: str | None = None,
    source: str = "chat",
    confidence: float = 0.5,
    message_id: int | None = None,
    cluster_topic: str | None = None,
    deduplicate: bool = True,
    embedding: list[float] | None = None,
    vector_store_obj: "VectorStore | None" = None,
    importance: float | None = None,
    decay_rate: float | None = None,
    memory_tier: int = 1,
    memory_type: str | None = None,
    pinned: bool = False,
    expires_at: datetime | None = None,
    use_count: int = 0,
) -> Memory | None:
    """
    Добавляет факт в память с дедупликацией.

    Два уровня дедупликации (при deduplicate=True):
      1. SHA256 хеш — точные повторы.
      2. Если передан embedding + vector_store_obj — семантическая
         дедупликация через Qdrant с динамическим порогом:
           - 0.92 — тот же source, возраст <7 дней (строже)
           - 0.78 — разные source (мягче)
           - 0.85 — остальные случаи

    При обнаружении дубликата повышает confidence (вес от source)
    и times_mentioned. Если факт содержит временные маркеры
    ("сейчас", "раньше", "уже не", "больше не", "перестал") —
    всегда создаётся новая запись.
    Если embedding передан, индексирует факт в Qdrant для будущих проверок.
    """
    from src.core.actions.stats_cache import invalidate

    fact = fact.strip()
    if len(fact) < 3:
        return None

    # Хеш для дедупликации (первые 64 бита SHA256)
    emb_hash = hashlib.sha256(fact.lower().strip().encode()).hexdigest()[:16]

    # Вес source для повышения confidence при мерже
    source_weight = {"chat": 0.3, "user": 0.6, "weekly": 0.15}.get(source, 0.3)

    # Временные маркеры — не мерджим, создаём как новый факт
    temporal_markers = {"сейчас", "раньше", "уже не", "больше не", "перестал"}
    has_temporal_marker = any(m in fact.lower() for m in temporal_markers)

    if deduplicate and not has_temporal_marker:
        # --- Уровень 1: SHA256 хеш (точные повторы) ---
        result = await session.execute(
            select(Memory)
            .where(
                Memory.user_id == user.id,
                Memory.embedding_hash == emb_hash,
            )
            .limit(1)
        )
        existing = result.scalar_one_or_none()
        if existing:
            existing.times_mentioned = (existing.times_mentioned or 1) + 1
            existing.confidence = min(1.0, existing.confidence + source_weight)
            existing.updated_at = datetime.now(timezone.utc)
            if sentiment and existing.sentiment != sentiment:
                existing.sentiment = "contradictory"  # маркируем противоречие
            await session.flush()
            await invalidate("mem_")
            return existing

        # --- Уровень 2: семантическая дедупликация через Qdrant ---
        if embedding is not None and vector_store_obj is not None:
            # Проверяем кэш эмбеддингов (на случай если embed уже закэширован)

            # Ищем кандидатов с запасом (порог 0.7)
            similar = await vector_store_obj.search_similar_memories(
                user_id=user.id,
                embedding=embedding,
                threshold=0.7,
                limit=3,
            )
            if similar:
                best = similar[0]
                existing = await session.get(Memory, best["memory_id"])
                if existing and existing.user_id == user.id:
                    # Динамический порог
                    now = datetime.now(timezone.utc)
                    age_days = (
                        (now - existing.created_at).days if existing.created_at else 999
                    )
                    same_source = existing.source == source
                    if same_source and age_days < 7:
                        dyn_threshold = 0.92
                    elif not same_source:
                        dyn_threshold = 0.78
                    else:
                        dyn_threshold = 0.85

                    if best["score"] >= dyn_threshold:
                        existing.times_mentioned = (existing.times_mentioned or 1) + 1
                        existing.confidence = min(
                            1.0, existing.confidence + source_weight
                        )
                        existing.updated_at = now
                        if sentiment and existing.sentiment != sentiment:
                            existing.sentiment = "contradictory"
                        await session.flush()
                        await invalidate("mem_")
                        return existing

    mem = Memory(
        user_id=user.id,
        contact_id=contact_id,
        fact=fact,
        sentiment=sentiment,
        source=source,
        confidence=confidence,
        times_mentioned=1,
        message_id=message_id,
        is_active=True,
        cluster_topic=cluster_topic,
        embedding_hash=emb_hash,
        importance=importance if importance is not None else 0.5,
        decay_rate=decay_rate if decay_rate is not None else 0.07,
        memory_tier=memory_tier,
        memory_type=memory_type,
        pinned=pinned,
        expires_at=expires_at,
        use_count=use_count,
    )
    session.add(mem)
    await session.flush()

    # Auto-link: connect to related facts via keyword overlap
    await _auto_link_memory(session, user, mem)

    try:
        from src.core.infra.hooks import hooks

        # Look up contact_name for hook callback
        contact_name: str | None = None
        if contact_id is not None:
            try:
                contact_result = await session.execute(
                    select(Contact.display_name).where(
                        Contact.user_id == user.id,
                        Contact.peer_id == contact_id,
                    )
                )
                contact_name = contact_result.scalar_one_or_none()
            except Exception:
                contact_name = None

        await hooks.emit(
            "on_memory_saved",
            memory_id=mem.id,
            fact=fact,
            user_id=user.telegram_id,
            contact_id=contact_id,
            contact_name=contact_name,
            confidence=confidence,
        )
    except Exception:
        pass  # hooks are optional, never break core flow

    # Индексируем эмбеддинг в Qdrant для будущей дедупликации
    if embedding is not None and vector_store_obj is not None:
        try:
            await vector_store_obj.upsert_memory(
                memory_id=mem.id,
                user_id=user.id,
                contact_id=contact_id,
                fact=fact,
                embedding=embedding,
            )
        except Exception:
            logger.exception("Failed to index memory embedding in Qdrant")

    await invalidate("mem_")
    return mem


async def _auto_link_memory(session: AsyncSession, user, memory) -> None:
    """Auto-link new fact to related facts via keyword overlap (no LLM).

    Finds candidate facts for the same contact, computes keyword overlap
    (words >= 4 chars), and creates a MemoryLink when overlap >= 2.
    Reuses the existing link_memories() for bidirectional link creation.
    """
    if not memory.fact or not memory.is_active:
        return

    # Keywords = words with 4+ characters
    words = {w.lower() for w in memory.fact.split() if len(w) >= 4}
    if len(words) < 2:
        return

    # Find active facts for same contact (limit 30)
    candidates_q = (
        select(Memory)
        .where(
            Memory.user_id == user.id,
            Memory.is_active == True,
            Memory.id != memory.id,
            Memory.contact_id == memory.contact_id,
        )
        .limit(30)
    )
    result = await session.execute(candidates_q)
    candidates = result.scalars().all()

    links_added = 0
    for c in candidates:
        if not c.fact:
            continue
        c_words = {w.lower() for w in c.fact.split() if len(w) >= 4}
        overlap = len(words & c_words)
        if overlap >= 2:
            # Link weight: base 0.3 + 0.1 per shared keyword
            await link_memories(
                session,
                user,
                source_id=memory.id,
                target_id=c.id,
                relation_type="related",
                weight=0.3 + overlap * 0.1,
            )
            links_added += 1

    if links_added:
        logger.debug("Auto-linked %d facts to memory %d", links_added, memory.id)


async def list_memories(
    session: AsyncSession,
    user,
    *,
    contact_id: int | None = None,
    limit: int | None = None,
) -> list[Memory]:
    query = (
        select(Memory)
        .where(Memory.user_id == user.id)
        .order_by(Memory.created_at.desc())
    )
    if contact_id is not None:
        query = query.where(Memory.contact_id == contact_id)
    if limit is not None:
        query = query.limit(limit)
    result = await session.execute(query)
    return list(result.scalars().all())


async def delete_memory(session: AsyncSession, user, memory_id: int) -> bool:
    from src.core.actions.stats_cache import invalidate

    m = await session.get(Memory, memory_id)
    if m is None or m.user_id != user.id:
        return False
    await session.delete(m)
    await invalidate("mem_")
    await session.flush()
    return True


async def add_memory_candidate(
    session: AsyncSession,
    user,
    *,
    fact: str,
    contact_id: int | None = None,
    sentiment: str | None = None,
    memory_type: str | None = None,
    source: str = "chat",
    importance: float = 0.5,
    decay_rate: float = 0.07,
) -> MemoryCandidate:
    candidate = MemoryCandidate(
        user_id=user.id,
        contact_id=contact_id,
        fact=fact,
        sentiment=sentiment,
        memory_type=memory_type,
        source=source,
        importance=importance,
        decay_rate=decay_rate,
    )
    session.add(candidate)
    await session.flush()
    return candidate


async def list_memory_candidates(
    session: AsyncSession,
    user,
    limit: int = 20,
) -> list[MemoryCandidate]:
    result = await session.execute(
        select(MemoryCandidate)
        .where(MemoryCandidate.user_id == user.id)
        .order_by(MemoryCandidate.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def delete_memory_candidate(
    session: AsyncSession,
    user,
    candidate_id: int,
) -> bool:
    obj = await session.get(MemoryCandidate, candidate_id)
    if obj and obj.user_id == user.id:
        await session.delete(obj)
        return True
    return False


async def search_memories(
    session: AsyncSession,
    user,
    query: str,
    *,
    contact_id: int | None = None,
) -> list[Memory]:
    # Пробуем FTS5 сначала; если пусто — ILIKE fallback
    results = await search_memories_fts(session, user, query, contact_id=contact_id)
    if results:
        return results
    stmt = (
        select(Memory)
        .where(
            Memory.user_id == user.id,
            Memory.fact.icontains(query),
        )
        .order_by(Memory.created_at.desc())
    )
    if contact_id is not None:
        stmt = stmt.where(Memory.contact_id == contact_id)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def search_memories_fts(
    session: AsyncSession,
    user,
    query: str,
    *,
    contact_id: int | None = None,
    limit: int = 50,
) -> list[Memory]:
    """Полнотекстовый поиск по памяти через FTS5 с ранжированием по bm25().

    Использует _fts_query_for() для преобразования запроса в FTS5-safe формат.
    """
    fts_q = _fts_query_for(query)
    if not fts_q:
        return []

    base_sql = """
        SELECT m.id FROM memories_fts
        JOIN memories m ON m.id = memories_fts.rowid
        WHERE memories_fts MATCH :q AND m.user_id = :uid
    """
    if contact_id is not None:
        base_sql += " AND m.contact_id = :cid"
    base_sql += " ORDER BY bm25(memories_fts) LIMIT :lim"

    params = {"q": fts_q, "uid": user.id, "lim": limit}
    if contact_id is not None:
        params["cid"] = contact_id

    result = await session.execute(sql_text(base_sql), params)
    ids = [r[0] for r in result.fetchall()]
    if not ids:
        return []

    rows = await session.execute(select(Memory).where(Memory.id.in_(ids)))
    mem_map = {m.id: m for m in rows.scalars().all()}
    return [mem_map[mid] for mid in ids if mid in mem_map]


async def search_memories_fts_with_scores(
    session: AsyncSession,
    user,
    query: str,
    *,
    contact_id: int | None = None,
    limit: int = 20,
) -> list[tuple[int, float]]:
    """FTS5 keyword search on memories_fts returning (memory_id, bm25_score).

    Returns results sorted by BM25 rank (ascending — lower is better).
    This is the keyword counterpart to vector_store.search_similar_memories()
    for use in reciprocal rank fusion (RRF).
    """
    fts_query = _fts_query_for(query)
    if not fts_query:
        return []

    sql_parts = [
        "SELECT m.id, bm25(memories_fts) AS score",
        "FROM memories_fts",
        "JOIN memories m ON m.id = memories_fts.rowid",
        "WHERE memories_fts MATCH :q AND m.user_id = :uid",
    ]
    params: dict = {"q": fts_query, "uid": user.id}

    if contact_id is not None:
        sql_parts.append("AND m.contact_id = :cid")
        params["cid"] = contact_id

    sql_parts.append("ORDER BY score")
    sql_parts.append("LIMIT :lim")
    params["lim"] = limit

    sql = "\n".join(sql_parts)
    result = await session.execute(sql_text(sql), params)
    rows = result.all()

    # Return (memory_id, bm25_score) — lower BM25 = better match
    return [(int(r[0]), float(r[1])) for r in rows if r[1] is not None]


async def find_similar_memories(
    session: AsyncSession, user, fact: str, threshold: float = 0.7
) -> list[Memory]:
    """Поиск похожих фактов: пробуем FTS5, fallback на ILIKE."""
    from sqlalchemy import text as _sa_text

    # Try FTS5 first
    try:
        fts_terms = [
            w + "*"
            for w in fact.split()
            if len(w) > 1 and w.replace("_", "").replace("-", "").isalnum()
        ]
        if fts_terms:
            fts_q = " OR ".join(fts_terms)
            result = await session.execute(
                select(Memory)
                .where(
                    Memory.user_id == user.id,
                    _sa_text("memories_fts MATCH :q").bindparams(q=fts_q),
                )
                .limit(20)
            )
            results = list(result.scalars().all())
            if results:
                return results
    except Exception:
        pass  # FTS5 table may not exist or query invalid, fall through to ILIKE

    # ILIKE fallback
    words = [w for w in fact.lower().split() if len(w) > 2]
    if not words:
        return []
    conditions = [Memory.fact.icontains(w) for w in words[:5]]
    result = await session.execute(
        select(Memory).where(Memory.user_id == user.id, or_(*conditions))
    )
    return list(result.scalars().all())


async def get_memory_stats(session: AsyncSession, user) -> dict:
    """Статистика по памяти (кэшируется на 5 минут). SQL-агрегация вместо загрузки всех объектов."""
    from src.core.actions.stats_cache import get_cached, set_cache

    cache_key = f"mem_stats:{user.id}"
    cached = await get_cached(cache_key)
    if cached is not None:
        return cached

    # Скалярные агрегаты одним запросом
    r = await session.execute(
        select(
            func.count().label("total"),
            func.coalesce(
                func.sum(case((Memory.confidence >= 0.8, 1), else_=0)), 0
            ).label("high_confidence"),
            func.coalesce(
                func.sum(case((Memory.contact_id.isnot(None), 1), else_=0)), 0
            ).label("with_contact"),
        ).where(Memory.user_id == user.id, Memory.is_active)
    )
    row = r.one()

    # По тональности
    sent_rows = (
        await session.execute(
            select(
                func.coalesce(Memory.sentiment, "neutral").label("sentiment"),
                func.count().label("cnt"),
            )
            .where(Memory.user_id == user.id, Memory.is_active)
            .group_by(func.coalesce(Memory.sentiment, "neutral"))
        )
    ).all()
    by_sentiment = {sr.sentiment: sr.cnt for sr in sent_rows}

    # По источникам
    src_rows = (
        await session.execute(
            select(Memory.source, func.count().label("cnt"))
            .where(Memory.user_id == user.id, Memory.is_active)
            .group_by(Memory.source)
        )
    ).all()
    by_source = {sr.source: sr.cnt for sr in src_rows}

    # По уровням памяти
    tier_rows = (
        await session.execute(
            select(
                Memory.memory_tier.label("tier"),
                func.count().label("cnt"),
            )
            .where(Memory.user_id == user.id, Memory.is_active)
            .group_by(Memory.memory_tier)
        )
    ).all()
    by_tier = {f"tier_{tr.tier}": tr.cnt for tr in tier_rows}

    stats = {
        "total": row.total,
        "by_sentiment": by_sentiment,
        "by_source": by_source,
        "by_tier": by_tier,
        "high_confidence": row.high_confidence,
        "with_contact": row.with_contact,
    }
    await set_cache(cache_key, stats)
    return stats


async def upsert_memory_cluster(
    session: AsyncSession,
    user,
    topic: str,
    *,
    summary: str | None = None,
    fact_count: int | None = None,
) -> MemoryCluster:
    """Создаёт или возвращает существующий кластер по теме."""
    result = await session.execute(
        select(MemoryCluster).where(
            MemoryCluster.user_id == user.id,
            MemoryCluster.topic == topic.lower().strip(),
        )
    )
    cluster = result.scalar_one_or_none()
    if cluster is None:
        cluster = MemoryCluster(user_id=user.id, topic=topic.lower().strip())
        session.add(cluster)
    if summary is not None:
        cluster.summary = summary
    if fact_count is not None:
        cluster.fact_count = fact_count
    await session.flush()
    return cluster


async def list_memory_clusters(session: AsyncSession, user) -> list[MemoryCluster]:
    """Список кластеров памяти."""
    result = await session.execute(
        select(MemoryCluster)
        .where(MemoryCluster.user_id == user.id)
        .order_by(MemoryCluster.fact_count.desc())
    )
    return list(result.scalars().all())


async def add_member(
    session: AsyncSession,
    user_id: int,
    memory_id: int,
    cluster_id: int,
    score: float = 0.5,
) -> None:
    """Добавляет факт в кластер."""
    m = MemoryClusterMember(
        user_id=user_id,
        memory_id=memory_id,
        cluster_id=cluster_id,
        relevance_score=score,
    )
    session.add(m)
    await session.flush()


async def get_cluster_members(
    session: AsyncSession,
    user,
    cluster_id: int,
    limit: int = 20,
) -> list[Memory]:
    """Факты кластера, отсортированы по relevance_score."""
    q = (
        select(Memory)
        .join(MemoryClusterMember, Memory.id == MemoryClusterMember.memory_id)
        .where(
            MemoryClusterMember.cluster_id == cluster_id,
            MemoryClusterMember.user_id == user.id,
            Memory.is_active,
        )
        .order_by(MemoryClusterMember.relevance_score.desc())
        .limit(limit)
    )
    r = await session.execute(q)
    return list(r.scalars().all())


async def list_clusters_for_contact(
    session: AsyncSession,
    user,
    contact_id: int | None = None,
) -> list:
    """Кластеры для контакта (или общие)."""
    q = (
        select(
            MemoryCluster,
            func.count(distinct(MemoryClusterMember.memory_id)).label("fact_count"),
        )
        .join(
            MemoryClusterMember,
            MemoryCluster.id == MemoryClusterMember.cluster_id,
        )
        .join(Memory, Memory.id == MemoryClusterMember.memory_id)
        .where(
            MemoryCluster.user_id == user.id,
            Memory.is_active,
        )
    )
    if contact_id is not None:
        q = q.where(Memory.contact_id == contact_id)
    q = (
        q.group_by(MemoryCluster.id)
        .order_by(func.count(distinct(MemoryClusterMember.memory_id)).desc())
        .limit(10)
    )
    r = await session.execute(q)
    return list(r.all())


async def link_memories(
    session: AsyncSession,
    user,
    source_id: int,
    target_id: int,
    weight: float = 0.5,
    relation_type: str | None = None,
) -> MemoryLink | None:
    """Создать/обновить связь между фактами памяти (many-to-many)."""

    # Проверить что оба факта принадлежат пользователю
    result = await session.execute(
        select(Memory).where(
            Memory.id.in_([source_id, target_id]), Memory.user_id == user.id
        )
    )
    if len(result.scalars().all()) < 2:
        return None  # один из фактов не найден или чужой

    # Проверить существующую связь
    existing = await session.execute(
        select(MemoryLink).where(
            MemoryLink.user_id == user.id,
            MemoryLink.source_id == source_id,
            MemoryLink.target_id == target_id,
        )
    )
    existing = existing.scalar_one_or_none()
    if existing:
        existing.weight = weight
        if relation_type:
            existing.relation_type = relation_type
        await session.flush()
        return existing

    # Создать новую + обратную
    link = MemoryLink(
        user_id=user.id,
        source_id=source_id,
        target_id=target_id,
        weight=weight,
        relation_type=relation_type,
    )
    session.add(link)

    # Обратная связь (если не дубль)
    reverse_check = await session.execute(
        select(MemoryLink).where(
            MemoryLink.user_id == user.id,
            MemoryLink.source_id == target_id,
            MemoryLink.target_id == source_id,
        )
    )
    if not reverse_check.scalar_one_or_none():
        rev = MemoryLink(
            user_id=user.id,
            source_id=target_id,
            target_id=source_id,
            weight=weight,
            relation_type=relation_type,
        )
        session.add(rev)

    await session.flush()
    return link


async def unlink_memories(
    session: AsyncSession, user, source_id: int, target_id: int
) -> None:
    """Удалить связь между фактами (в обе стороны)."""
    from sqlalchemy import and_, or_

    from sqlalchemy import delete as sa_delete

    await session.execute(
        sa_delete(MemoryLink).where(
            MemoryLink.user_id == user.id,
            or_(
                and_(
                    MemoryLink.source_id == source_id,
                    MemoryLink.target_id == target_id,
                ),
                and_(
                    MemoryLink.source_id == target_id,
                    MemoryLink.target_id == source_id,
                ),
            ),
        )
    )
    await session.flush()


async def get_linked_memories(
    session: AsyncSession, user, memory_id: int, limit: int = 10
) -> list[dict]:
    """Получить связанные факты с весами."""
    result = await session.execute(
        select(Memory, MemoryLink.weight, MemoryLink.relation_type)
        .join(MemoryLink, MemoryLink.target_id == Memory.id)
        .where(
            MemoryLink.source_id == memory_id,
            MemoryLink.user_id == user.id,
            Memory.is_active,
        )
        .order_by(MemoryLink.weight.desc())
        .limit(limit)
    )
    rows = result.all()
    linked: list[dict] = []
    for mem, weight, rel_type in rows:
        linked.append({"memory": mem, "weight": weight, "relation_type": rel_type})
    return linked


async def get_memory_graph(
    session: AsyncSession,
    user,
    memory_id: int,
    max_depth: int = 3,
    max_nodes: int = 20,
) -> list[dict]:
    """Строит граф связанных фактов BFS от memory_id.

    Оптимизация: вместо N запросов (на каждый узел) делаем 2 запроса:
    1) все MemoryLink пользователя → строим adjacency dict
    2) все Memory для посещённых ID → batch load
    """
    # ── Phase 1: Load ALL MemoryLinks for this user in ONE query ──────
    rows = (
        await session.execute(
            select(
                MemoryLink.source_id,
                MemoryLink.target_id,
                MemoryLink.weight,
                MemoryLink.relation_type,
            )
            .where(MemoryLink.user_id == user.id)
            .order_by(MemoryLink.weight.desc())
        )
    ).all()

    # Build in-memory adjacency dict: source_id -> [(target_id, weight, rel_type), ...]
    # Already sorted by weight DESC from the DB query
    adj: dict[int, list[tuple[int, float, str | None]]] = {}
    for source_id, target_id, weight, relation_type in rows:
        adj.setdefault(source_id, []).append((target_id, weight, relation_type))

    # ── Phase 2: BFS walk using the in-memory adjacency dict ─────────
    visited: set[int] = set()
    graph: list[dict] = []
    queue: list[tuple[int, int]] = [(memory_id, 0)]
    while queue and len(visited) < max_nodes:
        mid, depth = queue.pop(0)
        if mid in visited or depth > max_depth:
            continue
        visited.add(mid)
        if depth > 0:  # не добавляем корневой узел в граф, только соседей
            # Memory будет загружен в Phase 3 (batch)
            graph.append({"memory_id": mid, "depth": depth})
        if depth < max_depth:
            # adj.get(mid, []) уже отсортирован по weight DESC из Phase 1
            for target_id, weight, rel_type in adj.get(mid, [])[:10]:
                if target_id not in visited:
                    queue.append((target_id, depth + 1))

    if not graph:
        return []

    # ── Phase 3: Load ALL needed Memory objects in ONE batch query ───
    mem_ids = {entry["memory_id"] for entry in graph}
    result = await session.execute(select(Memory).where(Memory.id.in_(mem_ids)))
    mem_lookup: dict[int, Memory] = {m.id: m for m in result.scalars().all()}

    # ── Phase 4: Assemble the graph from the lookup dict ──────────────
    for entry in graph:
        mid = entry.pop("memory_id")
        mem = mem_lookup.get(mid)
        if mem:
            entry["memory"] = mem
        # если memory удалена между Phase 2 и Phase 3 — пропускаем
        # (аналогично оригинальному поведению `if mem:`)

    return graph
