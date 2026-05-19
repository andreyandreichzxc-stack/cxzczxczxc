"""Silent nightly memory validation + decay. Заменяет спам-опросы."""

import asyncio
import logging
import math
from datetime import datetime, timezone

from sqlalchemy import select

from src.core.temporal_layers import classify_layer, get_layer_config
from src.core.timeutil import now_in_tz
from src.db.models import Memory
from src.db.repo import get_or_create_user
from src.db.session import get_session

logger = logging.getLogger(__name__)

_CHUNK = 100


async def memory_decay_loop(owner_id: int) -> None:
    """Фоновый цикл: раз в сутки в 03:00 — decay + silent validation."""
    last_run = None
    while True:
        try:
            async with get_session() as session:
                owner = await get_or_create_user(session, owner_id)
                tz_name = owner.settings.timezone if owner.settings else "UTC"
            now = now_in_tz(tz_name)
            if now.hour == 3 and last_run != now.date():
                last_run = now.date()
                decayed, closed = await _run_decay_and_validation(owner_id)
                if closed > 0:
                    logger.info(
                        "Memory decay done: %d closed, %d decayed", closed, decayed
                    )
            await asyncio.sleep(600)  # каждые 10 минут проверка
        except Exception as e:
            logger.error("Memory decay error: %s", e)
            await asyncio.sleep(3600)


async def _run_decay_and_validation(owner_id: int) -> tuple[int, int]:
    """Применяет decay к фактам чанками по 100, коммит после каждого чанка.
    Возвращает (decayed_count, closed_count).
    """
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        now_utc = datetime.now(timezone.utc)

        decayed_count = 0
        closed_count = 0
        total_processed = 0
        offset = 0

        while True:
            result = await session.execute(
                select(Memory)
                .where(Memory.user_id == owner.id, Memory.is_active == True)
                .order_by(Memory.id)
                .offset(offset)
                .limit(_CHUNK)
            )
            chunk = list(result.scalars().all())
            if not chunk:
                break

            for mem in chunk:
                # Decay
                if mem.validity_start and mem.decay_rate:
                    days = (now_utc - mem.validity_start).total_seconds() / 86400
                    if days > 0:
                        layer = mem.temporal_layer or classify_layer(mem.created_at)
                        cfg = get_layer_config(layer)
                        effective_rate = mem.decay_rate * cfg["decay_multiplier"]

                        # Базовый decay (экспоненциальный)
                        base_decay = 2.71828 ** (-effective_rate * days)

                        # Adaptive multiplier от use_count
                        use_mult = 1.0
                        if mem.use_count > 10:
                            use_mult = 0.5  # очень медленно
                        elif mem.use_count > 5:
                            use_mult = 0.7
                        elif mem.use_count > 0:
                            use_mult = 0.85
                        elif mem.use_count == 0 and days > 14:
                            use_mult = 1.3  # неиспользуемый + старый → быстрее

                        # Adaptive multiplier от memory_type
                        type_mult = 1.0
                        if mem.memory_type == "temporary":
                            type_mult = 3.0  # быстро протухает
                        elif mem.memory_type == "preference":
                            type_mult = 0.3  # почти не протухает
                        elif mem.memory_type == "personal":
                            type_mult = 0.5
                        elif mem.memory_type == "contact_fact":
                            type_mult = 0.8

                        new_conf = mem.confidence * (base_decay * use_mult * type_mult)

                        if new_conf < 0.2:  # забылся
                            mem.is_active = False
                            mem.validity_end = now_utc
                            mem.confidence = new_conf
                            closed_count += 1
                        elif new_conf < mem.confidence * 0.7:  # значительно затух
                            mem.confidence = new_conf
                            decayed_count += 1

                await session.flush()

            await session.commit()
            total_processed += len(chunk)
            offset += _CHUNK

        if closed_count > 0:
            # Уведомление о ночной очистке
            from src.core.notifier import notifier

            await notifier.notify(
                f"🧠🌙 <b>Ночная очистка памяти:</b> {closed_count} фактов устарело и закрыто.\n"
                f"Обработано: {total_processed} активных фактов."
            )

    from src.core.stats_cache import invalidate

    await invalidate("mem_")
    return decayed_count, closed_count
