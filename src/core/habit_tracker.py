"""Habit Tracker — находит повторяющиеся факты и определяет привычки."""

import asyncio
import logging
from collections import defaultdict
from datetime import date, datetime, timezone
from typing import Any

from src.core.notification_queue import notification_queue
from src.db.models import Notification
from src.core.timeutil import now_in_tz
from src.db.repo import get_or_create_user, list_memories
from src.db.session import get_session

logger = logging.getLogger(__name__)

# Стоп-слова для фильтрации тем
STOP_WORDS: set[str] = {
    "я",
    "он",
    "она",
    "мы",
    "ты",
    "мне",
    "был",
    "была",
    "было",
    "это",
    "что",
    "как",
    "не",
    "на",
    "в",
    "с",
    "по",
    "из",
    "к",
    "у",
    "за",
    "от",
    "до",
    "для",
    "они",
    "она",
    "оно",
    "его",
    "её",
    "ее",
    "их",
    "нет",
    "да",
    "и",
    "а",
    "но",
    "или",
}


def extract_keywords(fact: str) -> set[str]:
    """Извлекает значимые слова из факта."""
    words = fact.lower().split()
    return {
        w.strip(".,!?:;()[]\"'«»") for w in words if len(w) > 3 and w not in STOP_WORDS
    }


def find_habit_candidates(
    memories: list,
    min_occurrences: int = 3,
    min_weeks: int = 2,
) -> list[dict[str, Any]]:
    """
    Ищет повторяющиеся факты с похожими ключевыми словами.

    Возвращает список словарей:
    [{topic, count, days_of_week, times_of_day, first_seen, last_seen,
      consistency_score, examples}]
    """
    # Группируем факты по темам (пересечение ключевых слов >= 40 %)
    clusters: list[dict[str, Any]] = []

    for m in memories:
        if not m.fact or not m.created_at:
            continue
        kw = extract_keywords(m.fact)
        if len(kw) < 2:
            continue

        matched = False
        for cluster in clusters:
            overlap = len(kw & cluster["keywords"])
            if overlap >= 2 and overlap / max(len(kw), len(cluster["keywords"])) >= 0.4:
                cluster["keywords"] |= kw
                cluster["facts"].append(m.fact[:80])
                cluster["dates"].append(m.created_at)
                matched = True
                break

        if not matched:
            clusters.append(
                {
                    "keywords": kw,
                    "facts": [m.fact[:80]],
                    "dates": [m.created_at],
                }
            )

    # Фильтруем привычки: минимум N повторений и разброс по неделям
    habits: list[dict[str, Any]] = []
    for c in clusters:
        if len(c["dates"]) < min_occurrences:
            continue
        dates = sorted(c["dates"])
        span_days = (dates[-1] - dates[0]).days
        if span_days < min_weeks * 7:
            continue

        # Дни недели и часы
        wdays: dict[int, int] = defaultdict(int)
        for d in dates:
            wdays[d.weekday()] += 1

        top_days = sorted(wdays.items(), key=lambda x: x[1], reverse=True)[:3]
        DAY_NAMES = ["ПН", "ВТ", "СР", "ЧТ", "ПТ", "СБ", "ВС"]
        day_str = ", ".join(DAY_NAMES[d] for d, cnt in top_days if cnt >= 2)

        # Consistency score (0-1): регулярность по дням недели
        max_day_count = max(wdays.values()) if wdays else 1
        consistency = max_day_count / len(dates) if dates else 0

        # Примеры фактов (последние 3)
        examples = c["facts"][-3:]

        # Тема — первое значимое слово
        topic = next(iter(c["keywords"]), "?").capitalize()

        habits.append(
            {
                "topic": topic,
                "count": len(c["dates"]),
                "days": day_str or "нерегулярно",
                "first_seen": dates[0],
                "last_seen": dates[-1],
                "consistency": round(consistency, 2),
                "examples": examples,
            }
        )

    habits.sort(key=lambda h: h["consistency"] * h["count"], reverse=True)
    return habits[:5]


def format_habits(habits: list[dict[str, Any]]) -> str:
    """Форматирует привычки для отображения."""
    if not habits:
        return "🔍 Привычек пока не обнаружено. Нужно больше данных."

    now = datetime.now(timezone.utc)
    lines = ["<b>🧠 Обнаруженные привычки:</b>", ""]
    for h in habits:
        topic = h["topic"]
        count = h["count"]
        days = h["days"]
        cons = h["consistency"]

        span = h["last_seen"] - h["first_seen"]
        weeks = max(1, span.days // 7)

        bar = "█" * int(cons * 10) + "░" * (10 - int(cons * 10))

        if cons >= 0.5:
            emoji = "💪"
            status = "формируется"
        else:
            emoji = "🌱"
            status = "намечается"

        lines.append(f"{emoji} <b>{topic}</b>: {count} раз за {weeks} нед ({days})")
        lines.append(f"   Регулярность: [{bar}] {status}")
        if h["examples"]:
            example = h["examples"][-1][:60]
            lines.append(f"   Пример: «{example}»")
        lines.append("")

    lines.append("<i>Используй /habits для обновления</i>")
    return "\n".join(lines)


async def habit_tracker_loop(owner_id: int) -> None:
    """Фоновый цикл: поиск привычек раз в неделю (ВС 18:00)."""
    last_run: date | None = None
    while True:
        try:
            async with get_session() as session:
                owner = await get_or_create_user(session, owner_id)
                tz_name = owner.settings.timezone if owner.settings else "UTC"

            now = now_in_tz(tz_name)
            # Воскресенье 18:00, не чаще раза в день
            if now.weekday() == 6 and now.hour == 18 and last_run != now.date():
                last_run = now.date()
                async with get_session() as session:
                    owner = await get_or_create_user(session, owner_id)
                    memories = await list_memories(session, owner)
                    active = [m for m in memories if m.is_active and m.created_at]
                    habits = find_habit_candidates(active)
                if habits:
                    text = format_habits(habits)
                    await notification_queue.enqueue(
                        topic="habits",
                        text=text,
                        priority=Notification.PRIORITY_LOW,
                    )
        except Exception as e:
            logger.exception("Habit tracker error: %s", e)
        await asyncio.sleep(3600)  # проверять каждый час
