"""Раз в N часов проверяет актуальность старых фактов из памяти."""

import asyncio
import logging
from datetime import datetime, timedelta

from src.config import settings as app_settings
from src.db.repo import get_or_create_user, list_memories
from src.db.session import get_session
from src.core.notifier import notifier

logger = logging.getLogger(__name__)

CHECK_INTERVAL_SEC = 6 * 3600  # раз в 6 часов
STALE_AFTER_DAYS = 1.5  # факты старше 1.5 дней считаем устаревшими


async def memory_checker_loop() -> None:
    while True:
        try:
            cutoff = datetime.utcnow() - timedelta(days=STALE_AFTER_DAYS)
            async with get_session() as session:
                owner = await get_or_create_user(
                    session, app_settings.owner_telegram_id
                )
                memories = await list_memories(session, owner)

            stale = [
                m
                for m in memories
                if m.sentiment in ("negative", "contradictory")
                and m.created_at < cutoff
            ]

            if stale:
                questions = []
                for m in stale[:3]:  # не больше 3 вопросов за раз
                    questions.append(f"- Ты говорил: «{m.fact}» — это ещё актуально?")

                if questions:
                    await notifier.notify(
                        "🧠 <b>Проверка памяти</b>\n\n"
                        + "\n".join(questions)
                        + "\n\n<i>Ответь в свободной форме или вызови /memory чтобы обновить.</i>"
                    )
        except Exception:
            logger.exception("memory_checker tick failed")

        await asyncio.sleep(CHECK_INTERVAL_SEC)
