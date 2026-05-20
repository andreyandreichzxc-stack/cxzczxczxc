"""Semantic Neighbors — находит эмбеддинг-соседей для фактов памяти.

Использует Qdrant (memory_facts) для поиска семантически близких фактов.
"""

import logging

from src.db.models import Memory
from src.db.repo import get_contact, get_or_create_user, list_memories
from src.db.session import get_session
from src.llm.router import build_provider

logger = logging.getLogger(__name__)


async def get_neighbors(owner_id: int, memory_id: int, limit: int = 3) -> list[dict]:
    """Находит топ-N семантических соседей для факта памяти.

    Загружает факт из БД, получает его эмбеддинг через LLM-провайдера
    и ищет похожие в Qdrant. Возвращает список соседей без самого факта.

    Параметры:
        owner_id: Telegram ID владельца.
        memory_id: ID факта памяти в БД.
        limit: Сколько соседей вернуть (по умолчанию 3).

    Возвращает:
        Список словарей: {memory_id, fact, contact_name, similarity}.
    """
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        provider = await build_provider(session, owner)
        mem = await session.get(Memory, memory_id)
        if not mem or mem.user_id != owner.id:
            return []
        if not provider:
            return []

        try:
            embedding = await provider.embed(mem.fact)
        except Exception:
            logger.debug("Failed to embed for neighbors (memory_id=%d)", memory_id)
            return []

    from src.core.actions.vector_store import vector_store

    # Ищем похожие факты в Qdrant (+1 чтобы потом исключить сам факт)
    neighbors = await vector_store.search_similar_memories(
        user_id=owner.id,
        embedding=embedding,
        threshold=0.65,
        limit=limit + 1,
    )

    result: list[dict] = []
    async with get_session() as session2:
        for n in neighbors:
            nid = n.get("memory_id", 0)
            if nid == memory_id or nid == mem.id:
                continue
            contact_name = ""
            cid = n.get("contact_id")
            if cid:
                contact = await get_contact(session2, owner, cid)
                contact_name = contact.display_name if contact else ""
            result.append(
                {
                    "memory_id": nid,
                    "fact": (n.get("fact", "") or "")[:100],
                    "similarity": round(n.get("score", 0), 2),
                    "contact_name": contact_name,
                }
            )
            if len(result) >= limit:
                break
    return result


def format_neighbors(neighbors: list[dict], fact_text: str = "") -> str:
    """Форматирует соседей для показа в UI (HTML)."""
    if not neighbors:
        return ""
    lines = ["<b>🔗 Семантически близкие факты:</b>"]
    for n in neighbors:
        sim = n["similarity"]
        bar_count = max(1, min(5, int(sim * 5)))
        bar = "█" * bar_count + "░" * (5 - bar_count)
        contact = f" ({n['contact_name']})" if n["contact_name"] else ""
        lines.append(f"  [{bar}] {n['fact']}{contact}")
    return "\n".join(lines)


async def find_cross_contact_bridges(owner_id: int) -> list[dict]:
    """Находит факты из РАЗНЫХ контактов с похожим смыслом.

    Ищет кластеры семантически близких фактов с разными contact_id.
    Возвращает до 5 пар контактов, между которыми есть смысловая связь.

    Параметры:
        owner_id: Telegram ID владельца.

    Возвращает:
        Список словарей: {contact1, contact2, fact1, fact2, similarity}.
    """
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        memories = await list_memories(session, owner)
        active = [m for m in memories if m.is_active and m.contact_id]
        if len(active) < 10:
            return []

        provider = await build_provider(session, owner)
        if not provider:
            return []

    from src.core.actions.vector_store import vector_store

    bridges: list[dict] = []
    seen_pairs: set[tuple[int, int]] = set()

    for m1 in active[:50]:  # макс 50 для производительности
        try:
            emb = await provider.embed(m1.fact)
        except Exception:
            continue

        neighbors = await vector_store.search_similar_memories(
            user_id=owner.id,
            embedding=emb,
            threshold=0.75,
            limit=5,
        )
        for n in neighbors:
            nid = n.get("memory_id", 0)
            n_cid = n.get("contact_id", 0)
            if nid == m1.id or not n_cid:
                continue
            if m1.contact_id != n_cid:  # РАЗНЫЕ контакты! (both non-None: line 122+136)
                pair = (m1.contact_id, n_cid)  # type: ignore[arg-type]
                if pair not in seen_pairs:
                    seen_pairs.add(pair)  # type: ignore[arg-type]
                    async with get_session() as s:
                        c1 = await get_contact(s, owner, m1.contact_id)  # type: ignore[arg-type]
                        c2 = await get_contact(s, owner, n_cid)  # type: ignore[arg-type]
                        bridges.append(
                            {
                                "contact1": c1.display_name
                                if c1
                                else str(m1.contact_id),
                                "contact2": c2.display_name if c2 else str(n_cid),
                                "fact1": (m1.fact or "")[:80],
                                "fact2": (n.get("fact", "") or "")[:80],
                                "similarity": round(n.get("score", 0), 2),
                            }
                        )
                    if len(bridges) >= 5:
                        break
        if len(bridges) >= 5:
            break

    return bridges


def format_bridges(bridges: list[dict]) -> str:
    """Форматирует смысловые мосты между контактами для показа в UI (HTML)."""
    if not bridges:
        return ""
    lines: list[str] = ["<b>🌉 Смысловые мосты между контактами:</b>", ""]
    for b in bridges:
        lines.append(
            f"▸ <b>{b['contact1']}</b> ↔ <b>{b['contact2']}</b> (sim {b['similarity']})"
        )
        lines.append(f"  «{b['fact1']}»")
        lines.append(f"  «{b['fact2']}»")
        lines.append("")
    return "\n".join(lines)


async def cross_contact_insights(owner_id: int) -> list[str]:
    """Находит общие темы и паттерны между контактами.

    Анализирует факты всех контактов, ищет общие ключевые слова,
    пересекающиеся темы (теги), и повторяющиеся ситуации.

    Возвращает список строк-инсайтов вида:
    - "и Настя, и Артём упоминали дедлайны"
    - "у Насти и Лены общая тема: ремонт"
    """
    from collections import Counter

    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        memories = await list_memories(session, owner)
        active_with_contact = [m for m in memories if m.is_active and m.contact_id]

    if len(active_with_contact) < 5:
        return []

    # Группируем факты по контактам
    from collections import defaultdict

    by_contact: dict[int, list] = defaultdict(list)
    for m in active_with_contact:
        by_contact[m.contact_id].append(m)

    # Собираем ключевые слова для каждого контакта
    contact_keywords: dict[int, set[str]] = {}
    for cid, facts in by_contact.items():
        words: set[str] = set()
        for m in facts:
            # Извлекаем слова длиннее 3 букв
            tokens = [
                w.lower().strip(",.!?…():;-«»\"'")
                for w in m.fact.split()
                if len(w.strip(",.!?…():;-«»\"'")) > 3
            ]
            words.update(tokens)
        contact_keywords[cid] = words

    # Ищем пересечения
    insights: list[str] = []
    contact_ids = list(contact_keywords.keys())

    for i in range(len(contact_ids)):
        for j in range(i + 1, len(contact_ids)):
            cid1 = contact_ids[i]
            cid2 = contact_ids[j]
            common = contact_keywords[cid1] & contact_keywords[cid2]
            if len(common) >= 2:
                # Получаем имена контактов
                async with get_session() as s:
                    owner = await get_or_create_user(s, owner_id)
                    c1 = await get_contact(s, owner, cid1)
                    c2 = await get_contact(s, owner, cid2)
                name1 = c1.display_name if c1 else str(cid1)
                name2 = c2.display_name if c2 else str(cid2)
                common_words = ", ".join(sorted(common)[:4])
                insights.append(
                    f"🔗 И <b>{name1}</b>, и <b>{name2}</b> упоминали: {common_words}"
                )
                if len(insights) >= 5:
                    break
        if len(insights) >= 5:
            break

    return insights


def format_cross_insights(insights: list[str]) -> str:
    """Форматирует кросс-контактные инсайты для показа в UI."""
    if not insights:
        return ""
    lines = ["<b>🔗 Связи между контактами:</b>"]
    for ins in insights:
        lines.append(f"  {ins}")
    return "\n".join(lines)
