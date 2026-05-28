"""Temporal Memory Layers — факты мигрируют между временными слоями."""

import logging
from datetime import datetime, timezone

from sqlalchemy import or_, select

from src.db.models import Memory
from src.db.repo import get_or_create_user, list_memories
from src.config import settings
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


def utc_naive(dt: datetime) -> datetime:
    """Return a UTC-naive datetime compatible with SQLite DateTime columns."""
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def utcnow_naive() -> datetime:
    """Current UTC time as a naive datetime for DB comparisons."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def compute_retention(
    memory,
    now: datetime | None = None,
    decay_base: float = 0.07,
    access_weight: float = 0.5,
) -> float:
    """Compute Ebbinghaus retention score for a memory fact.

    retention(t) = e^(-decay_rate * t * access_boost)

    where:
    - t = days since last_used_at (or created_at if never recalled)
    - decay_rate = memory.decay_rate or decay_base
    - access_boost = 1.0 / (1.0 + access_weight * log(1 + use_count))

    Returns value in [0, 1]. Higher = better retained.
    """
    import math

    if now is None:
        now = utcnow_naive()

    # Reference time: last recall or creation
    ref_time = memory.last_used_at or memory.created_at
    if ref_time is None:
        return 1.0  # brand new fact

    t_days = max(0.0, (utc_naive(now) - utc_naive(ref_time)).total_seconds() / 86400.0)

    decay_rate = memory.decay_rate if memory.decay_rate is not None else decay_base
    use_count = memory.use_count or 0

    # Access boost: each recall slows down forgetting
    access_boost = 1.0 / (1.0 + access_weight * math.log(1 + use_count))

    retention = math.exp(-decay_rate * t_days * access_boost)
    return max(0.0, min(1.0, retention))


def classify_layer(memory_or_dt, now: datetime | None = None, **kwargs) -> str:
    """Определяет временной слой факта по retention score (Ebbinghaus).

    Accepts either a Memory object (with decay_rate, use_count, last_used_at)
    or a datetime for backward compatibility.
    """
    if now is None:
        now = utcnow_naive()

    # Backward compat: if passed a datetime, use age-based classification
    if isinstance(memory_or_dt, datetime):
        age_days = (utc_naive(now) - utc_naive(memory_or_dt)).days
        if age_days <= 7:
            return "recent"
        elif age_days <= 30:
            return "medium"
        return "longterm"

    # New: retention-based classification
    retention = compute_retention(memory_or_dt, now, **kwargs)
    if retention >= 0.7:
        return "recent"
    elif retention >= 0.3:
        return "medium"
    return "longterm"


def get_layer_config(layer: str) -> dict:
    """Конфиг слоя."""
    return LAYER_CONFIG.get(layer, LAYER_CONFIG["recent"])


async def update_temporal_layers(owner_id: int) -> int:
    """Обновляет temporal_layer для ВСЕХ фактов памяти."""
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        memories = await list_memories(session, owner)
        now = utcnow_naive()
        updated = 0
        for m in memories:
            if m.created_at and m.is_active:
                new_layer = classify_layer(m, now)
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
        now = utcnow_naive()
        for m in active:
            layer = m.temporal_layer or (
                classify_layer(m, now) if m.created_at else "recent"
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
    now = utcnow_naive()
    conditions = [
        Memory.user_id == owner.id,
        Memory.is_active,
        or_(Memory.expires_at.is_(None), Memory.expires_at > now),
    ]
    if contact_id:
        conditions.append(Memory.contact_id == contact_id)
    result = await session.execute(
        select(Memory)
        .where(*conditions)
        .order_by(Memory.confidence.desc(), Memory.created_at.desc())
    )
    all_facts = list(result.scalars().all())
    buckets: dict[str, list] = {"recent": [], "medium": [], "longterm": []}
    # Сортируем: pinned всегда первыми, затем по retention, затем use_count, затем confidence
    all_facts.sort(
        key=lambda m: (
            m.pinned,
            compute_retention(m, now),
            m.use_count or 0,
            m.confidence or 0,
        ),
        reverse=True,
    )

    for m in all_facts:
        layer = m.temporal_layer or (
            classify_layer(m, now) if m.created_at else "recent"
        )
        if len(buckets[layer]) < LAYER_CONFIG[layer]["max_facts_in_prompt"]:
            buckets[layer].append(m)
    picked: list = []
    for layer in ["recent", "medium", "longterm"]:
        picked.extend(buckets[layer])
        if len(picked) >= total_limit:
            break
    if picked:
        await session.flush()
    return picked[:total_limit]


async def temporal_migration_loop(owner_id: int) -> None:
    """Фоновый цикл: раз в час обновляет слои."""
    import asyncio

    while True:
        try:
            await update_temporal_layers(owner_id)
        except Exception:
            logger.exception("Temporal migration error")
        await asyncio.sleep(settings.temporal_migration_interval_sec)


from functools import partial
from src.core.infra.task_manager import task_manager

task_manager.register(
    "temporal-migration", partial(temporal_migration_loop, settings.owner_telegram_id)
)
