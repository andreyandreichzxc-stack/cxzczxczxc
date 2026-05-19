"""Memory Chain Narrative — строит связные рассказы из связанных фактов памяти."""

import logging
from datetime import datetime

from src.db.repo import (
    get_contact,
    get_linked_memories,
    get_or_create_user,
    list_memories,
)
from src.db.session import get_session

logger = logging.getLogger(__name__)

RELATION_EMOJI = {
    "cause": "🎯",
    "effect": "⚡",
    "contradicts": "⚠️",
    "supports": "✅",
    "continues": "➡️",
    "example_of": "📌",
    None: "•",
}

RELATION_WORD = {
    "cause": "потому что",
    "effect": "из-за этого",
    "contradicts": "но",
    "supports": "и это подтверждает что",
    "continues": "затем",
    "example_of": "например",
    None: "",
}


async def build_chain(
    session, owner, memory_id: int, max_depth: int = 10
) -> list[dict]:
    """
    Строит цепочку связанных фактов от memory_id в обе стороны.
    Возвращает список фактов в хронологическом порядке.
    Каждый элемент: {memory_id, fact, sentiment, relation_type, related_to, created_at}
    """
    seen = set()
    chain = []
    queue = [memory_id]
    while queue and len(chain) < max_depth:
        mid = queue.pop(0)
        if mid in seen:
            continue
        seen.add(mid)
        linked = await get_linked_memories(session, owner, mid, limit=5)
        for item in linked:
            m = item["memory"]
            if m.id not in seen:
                queue.append(m.id)
                chain.append(
                    {
                        "memory_id": m.id,
                        "fact": m.fact,
                        "sentiment": m.sentiment,
                        "relation_type": item.get("relation_type"),
                        "related_to": mid,
                        "weight": item.get("weight", 0.5),
                        "created_at": m.created_at,
                    }
                )
    return sorted(chain, key=lambda x: x["created_at"] or datetime.min)


async def build_chain_narrative(contact_id: int, owner_id: int) -> str | None:
    """
    Строит связный рассказ обо всех фактах контакта.
    Возвращает HTML-строку или None если не хватает данных.
    """
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        memories = await list_memories(session, owner, contact_id=contact_id)
        if len(memories) < 3:
            return None
        contact = await get_contact(session, owner, contact_id)
        name = contact.display_name if contact else str(contact_id)

        # Строим группы связанных фактов
        seen: set[int] = set()
        chains: list[list[dict]] = []

        # Начинаем с фактов у которых есть relation_type (корни цепочек)
        roots = [m for m in memories if m.relation_type and not m.related_memory_id]
        if not roots:
            roots = memories[:1]

        for root in roots[:3]:  # макс 3 цепочки
            chain = await build_chain(session, owner, root.id, max_depth=8)
            if chain:
                chains.append(chain)
                for c in chain:
                    seen.add(c["memory_id"])

        # Формируем текст
        lines = [f"<b>📖 История отношений с {name}</b>", ""]
        for ci, chain in enumerate(chains):
            if ci > 0:
                lines.append("")
            lines.append(f"<b>Сюжет {ci + 1}:</b>")
            for item in chain:
                emoji = RELATION_EMOJI.get(item["relation_type"], "•")
                word = RELATION_WORD.get(item["relation_type"], "")
                prefix = f"  {emoji} {word} " if word else f"  {emoji} "
                lines.append(f"{prefix}{item['fact']}")

        # Одинокие факты (не в цепочках)
        orphans = [m for m in memories if m.id not in seen]
        if orphans and len(orphans) <= 10:
            lines.append("")
            lines.append("<b>Другие факты:</b>")
            for m in orphans[:5]:
                lines.append(f"  • {m.fact}")

        return "\n".join(lines)


def format_chain_compact(memories: list, contact_name: str = "") -> str:
    """Компактный формат цепочки для вставки в /threads или /chat."""
    if not memories:
        return ""
    name = contact_name or "контакта"
    lines = [f"<b>🔗 История с {name}:</b>"]
    for m in memories[:6]:
        emoji = RELATION_EMOJI.get(
            m.relation_type if hasattr(m, "relation_type") else None,
            "•",
        )
        lines.append(f"{emoji} {m.fact if hasattr(m, 'fact') else str(m)}")
    return "\n".join(lines)
