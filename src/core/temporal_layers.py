"""Temporal Memory Layers — факты мигрируют между временными слоями."""

import logging
from datetime import datetime, timezone

from sqlalchemy import select

from src.db.models import Memory
from src.db.repo import get_or_create_user, list_memories
from src.db.session import get_session

logger = logging.getLogger(__name__)

LAYER_CONFIG = {
    "recent": {
        "emoji": "🔥",
        "label": "Недавние",
        "max_days": 7,
        "decay_multiplier": 1.0,  # обычный decay
        "prompt_priority": 3,  # выше всех в промпте
        "max_facts_in_prompt": 5,
    },
    "medium": {
        "emoji": "🌗",
        "label": "Средние",
        "max_days": 30,
        "decay_multiplier": 0.7,  # медленнее decay
        "prompt_priority": 2,
        "max_facts_in_prompt": 3,
    },
    "longterm": {
        "emoji": "🏛️",
        "label": "Долгосрочные",
        "max_days": None,  # без ограничения
        "decay_multiplier": 0.3,  # очень медленный decay
        "prompt_priority": 1,
        "max_facts_in_prompt": 2,
    },
}


def classify_layer(created_at: datetime, now: datetime | None = None) -> str:
    """Определяет временной слой факта по дате создания."""
    if now is None:
        now = datetime.now(timezone.utc)
    age_days = (now - created_at).days
    if age_days <= 7:
        return "recent"
    elif age_days <= 30:
        return "medium"
    else:
        return "longterm"


def get_layer_config(layer: str) -> dict:
    """Конфиг слоя."""
    return LAYER_CONFIG.get(layer, LAYER_CONFIG["recent"])


async def update_temporal_layers(owner_id: int) -> int:
    """Обновляет temporal_layer для ВСЕХ фактов памяти."""
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        memories = await list_memories(session, owner)
        now = datetime.now(timezone.utc)
        updated = 0
        for m in memories:
            if m.created_at and m.is_active:
                new_layer = classify_layer(m.created_at, now)
                if m.temporal_layer != new_layer:
                    m.temporal_layer = new_layer
                    updated += 1
        if updated > 0:
            await session.commit()
            logger.info("Updated temporal layers: %d facts migrated", updated)
    return updated


async def get_layer_stats(owner_id: int) -> dict:
    """Статистика по временным слоям."""
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        memories = await list_memories(session, owner)
        active = [m for m in memories if m.is_active]
        stats: dict = {
            "recent": 0,
            "medium": 0,
            "longterm": 0,
            "total": len(active),
        }
        now = datetime.now(timezone.utc)
        for m in active:
            layer = m.temporal_layer or (
                classify_layer(m.created_at, now) if m.created_at else "recent"
            )
            stats[layer] += 1  # type: ignore[literal-required]
        return stats


def format_layer_stats(stats: dict) -> str:
    """Форматирует статистику слоёв."""
    total = stats["total"]
    if total == 0:
        return "🧠 Память пуста."
    lines = ["<b>🧠 Слои памяти:</b>"]
    for layer in ["recent", "medium", "longterm"]:
        cfg = LAYER_CONFIG[layer]
        count = stats[layer]
        pct = f" ({count / total * 100:.0f}%)" if total > 0 else ""
        bar = "█" * min(count, 20)
        lines.append(f"{cfg['emoji']} {cfg['label']}: {count}{pct} {bar}")
    return "\n".join(lines)


async def get_prompt_facts(
    session,
    owner,
    contact_id: int | None = None,
    total_limit: int = 8,
) -> list:
    """
    Возвращает факты для инжекции в промпт с учётом слоёв.
    Приоритет: recent (5) > medium (3) > longterm (2). Всего до total_limit.
    """
    conditions = [Memory.user_id == owner.id, Memory.is_active == True]
    if contact_id:
        conditions.append(Memory.contact_id == contact_id)
    result = await session.execute(
        select(Memory)
        .where(*conditions)
        .order_by(Memory.confidence.desc(), Memory.created_at.desc())
    )
    all_facts = list(result.scalars().all())
    now = datetime.now(timezone.utc)
    buckets: dict[str, list] = {"recent": [], "medium": [], "longterm": []}
    # Сортируем: pinned всегда первыми, затем по use_count, затем по confidence
    all_facts.sort(key=lambda m: (m.pinned, m.use_count, m.confidence), reverse=True)

    for m in all_facts:
        layer = m.temporal_layer or (
            classify_layer(m.created_at, now) if m.created_at else "recent"
        )
        if len(buckets[layer]) < LAYER_CONFIG[layer]["max_facts_in_prompt"]:
            buckets[layer].append(m)
    picked: list = []
    for layer in ["recent", "medium", "longterm"]:
        picked.extend(buckets[layer])
        if len(picked) >= total_limit:
            break
    return picked[:total_limit]


async def temporal_migration_loop(owner_id: int) -> None:
    """Фоновый цикл: раз в час обновляет слои."""
    import asyncio

    while True:
        try:
            await update_temporal_layers(owner_id)
        except Exception as e:
            logger.error("Temporal migration error: %s", e)
        await asyncio.sleep(3600)
