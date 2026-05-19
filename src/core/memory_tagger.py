"""Auto-tagging — LLM проставляет теги фактам памяти."""

import json
import logging

from src.db.repo import get_or_create_user, list_memories
from src.db.session import get_session
from src.llm.base import ChatMessage

logger = logging.getLogger(__name__)

DEFAULT_TAGS = [
    "работа",
    "семья",
    "здоровье",
    "деньги",
    "отношения",
    "поездки",
    "учеба",
    "хобби",
    "дом",
    "планы",
]

TAGGING_PROMPT = """Проставь теги для факта памяти. 
Доступные теги: {available_tags}
Можно выбрать 1-3 тега, только если они точно подходят.

Факт: {fact}

Верни ТОЛЬКО JSON: {{"tags": ["тег1", "тег2"]}}"""


async def tag_fact(
    provider, fact: str, available_tags: list[str] | None = None
) -> list[str]:
    """LLM проставляет теги для одного факта."""
    if not provider or not fact:
        return []
    tags_list = available_tags or DEFAULT_TAGS
    try:
        raw = await provider.chat(
            [
                ChatMessage(
                    role="system",
                    content="Ты — классификатор. Отвечай ТОЛЬКО JSON.",
                ),
                ChatMessage(
                    role="user",
                    content=TAGGING_PROMPT.format(
                        available_tags=", ".join(tags_list), fact=fact[:200]
                    ),
                ),
            ],
            heavy=False,
        )
    except Exception:
        logger.exception("Tagging LLM call failed for fact: %r", fact[:60])
        return []
    import re

    raw = raw.strip()
    raw = re.sub(r"^```(?:json|JSON)?\s*\n?", "", raw)
    raw = re.sub(r"\n?\s*```\s*$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    tag_list = data.get("tags", [])
    valid_set = set(tags_list or DEFAULT_TAGS)
    if isinstance(tag_list, list):
        return [t.strip().lower() for t in tag_list if t.strip().lower() in valid_set]
    return []


async def tag_all_untagged(owner_id: int) -> int:
    """Проставляет теги для всех нетэгированных фактов."""
    from src.llm.router import build_provider

    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        provider = await build_provider(session, owner)
        if not provider:
            return 0
        memories = await list_memories(session, owner)
        untagged = [
            m
            for m in memories
            if m.is_active and m.fact and not m.tags and len(m.fact) >= 10
        ]
        tagged = 0
        for m in untagged:
            tags = await tag_fact(provider, m.fact)
            if tags:
                m.tags = ",".join(tags)
                tagged += 1
        if tagged > 0:
            await session.commit()
        return tagged


async def tag_new_fact(provider, session, memory_id: int) -> None:
    """Тегирует один новый факт при сохранении."""
    from src.db.models import Memory

    mem = await session.get(Memory, memory_id)
    if mem and mem.fact and not mem.tags and len(mem.fact) >= 10:
        tags = await tag_fact(provider, mem.fact)
        if tags:
            mem.tags = ",".join(tags)
            await session.flush()


async def search_by_tag(session, owner, tag: str) -> list:
    """Поиск фактов по тегу."""
    from sqlalchemy import select

    from src.db.models import Memory

    result = await session.execute(
        select(Memory)
        .where(
            Memory.user_id == owner.id,
            Memory.is_active == True,
            Memory.tags.ilike(f"%{tag}%"),
        )
        .order_by(Memory.created_at.desc())
    )
    return list(result.scalars().all())


def format_tagged(memories: list, tag: str) -> str:
    """Форматирует факты по тегу."""
    if not memories:
        return f"🏷 Нет фактов с тегом «{tag}»."
    lines = [f"<b>🏷 Тег: {tag}</b> ({len(memories)} фактов)", ""]
    for m in memories[:15]:
        tags_str = f" [{m.tags}]" if m.tags else ""
        lines.append(f"• {m.fact[:100]}{tags_str}")
    return "\n".join(lines)


async def get_tag_stats(owner_id: int) -> dict:
    """Статистика по тегам."""
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        memories = await list_memories(session, owner)
        tag_counts: dict[str, int] = {}
        for m in memories:
            if m.is_active and m.tags:
                for t in m.tags.split(","):
                    t = t.strip().lower()
                    if t:
                        tag_counts[t] = tag_counts.get(t, 0) + 1
        return dict(sorted(tag_counts.items(), key=lambda x: -x[1]))


def format_tag_stats(stats: dict) -> str:
    """Форматирует статистику тегов."""
    if not stats:
        return ""
    lines = ["<b>🏷 Теги памяти:</b>"]
    for tag, count in list(stats.items())[:8]:
        lines.append(f"  #{tag}: {count}")
    return "\n".join(lines)
