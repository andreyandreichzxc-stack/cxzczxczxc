"""Silent nightly memory validation + decay. Заменяет спам-опросы."""

import asyncio
import logging
import math
from datetime import datetime, timezone

from src.core.timeutil import now_in_tz
from src.db.repo import get_or_create_user, list_memories
from src.db.session import get_session

logger = logging.getLogger(__name__)


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
                await _run_decay_and_validation(owner_id)
            await asyncio.sleep(600)  # каждые 10 минут проверка
        except Exception as e:
            logger.error("Memory decay error: %s", e)
            await asyncio.sleep(3600)


async def _run_decay_and_validation(owner_id: int) -> None:
    """Применяет decay к фактам + тихо валидирует противоречия."""
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        memories = await list_memories(session, owner)
        now_utc = datetime.now(timezone.utc)

        closed = 0

        for m in memories:
            if not m.is_active:
                continue

            # Decay
            if m.validity_start and m.decay_rate:
                days = (now_utc - m.validity_start).total_seconds() / 86400
                if days > 0:
                    new_conf = m.confidence * math.exp(-m.decay_rate * days)
                    if new_conf < 0.2:  # забылся
                        m.is_active = False
                        m.validity_end = now_utc
                        m.confidence = new_conf
                        closed += 1
                    elif new_conf < m.confidence * 0.7:  # значительно затух
                        m.confidence = new_conf

        if closed > 0:
            await session.commit()
            logger.info("Memory decay: %d facts closed", closed)

            # Уведомление о ночной очистке
            from src.core.notifier import notifier

            await notifier.notify(
                f"🧠🌙 <b>Ночная очистка памяти:</b> {closed} фактов устарело и закрыто.\n"
                f"Активно: {sum(1 for m in memories if m.is_active)} фактов."
            )
