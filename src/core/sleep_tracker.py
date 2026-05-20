"""Sleep pattern tracker — анализирует паттерны сна владельца."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from src.core.timeutil import now_in_tz
from src.db.repo import get_or_create_user
from src.config import settings
from src.db.session import get_session

logger = logging.getLogger(__name__)


async def sleep_tracker_loop(owner_id: int) -> None:
    """Фоновый цикл: каждые 15 минут проверяет паттерны сна."""
    from src.core.notification_queue import notification_queue
    from src.db.models import Notification

    _notified_for_date: str | None = (
        None  # дата (YYYY-MM-DD) последнего sleep-уведомления
    )
    while True:
        try:
            async with get_session() as session:
                owner = await get_or_create_user(session, owner_id)
                tz_name = owner.settings.timezone if owner.settings else "UTC"
                now = now_in_tz(tz_name)
                hour = now.hour

                # Ночной интервал 22:00 - 08:00
                is_night = hour >= 22 or hour < 8

                if is_night:
                    last_seen = owner.last_seen_online
                    if last_seen is not None:
                        dt_now_utc = datetime.now(timezone.utc)
                        offline_minutes = (dt_now_utc - last_seen).total_seconds() / 60
                        # Уже спит >60 минут подряд
                        if offline_minutes > 60 and owner.absence_status != "sleeping":
                            owner.absence_status = "sleeping"
                            owner.absence_message = f"Спит с {now.strftime('%H:%M')}"
                            await session.commit()
                            # Одно уведомление в начале сна (не чаще раза в день)
                            today_str = now.strftime("%Y-%m-%d")
                            if _notified_for_date != today_str:
                                _notified_for_date = today_str
                                await notification_queue.enqueue(
                                    topic="sleep",
                                    text=(
                                        "😴🌙 <b>Режим сна активирован</b>\n"
                                        "Авто-ответы будут говорить что ты спишь.\n"
                                        "Отключится автоматически утром."
                                    ),
                                    priority=Notification.PRIORITY_LOW,
                                )
                else:
                    # Дневное время — сброс sleeping статуса
                    if owner.absence_status == "sleeping":
                        owner.absence_status = None
                        owner.absence_message = None
                        await session.commit()
                        if _notified_for_date is not None:
                            await notification_queue.enqueue(
                                topic="sleep",
                                text=(
                                    "☀️ <b>Доброе утро!</b> Режим сна отключён. "
                                    "Авто-ответы вернулись в обычный режим."
                                ),
                                priority=Notification.PRIORITY_LOW,
                            )
                            _notified_for_date = None
            await asyncio.sleep(settings.sleep_tracker_check_sec)  # каждые 15 минут
        except Exception as e:
            logger.error("Sleep tracker error: %s", e)
            await asyncio.sleep(settings.sleep_tracker_fallback_sec)
