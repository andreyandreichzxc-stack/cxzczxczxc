"""Proactive pattern detection — находит закономерности в памяти и предлагает действия."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from src.core.notifier import notifier
from src.core.timeutil import now_in_tz
from src.db.repo import (
    get_contact,
    get_or_create_user,
    list_contacts,
    list_memories,
)
from src.db.session import get_session

logger = logging.getLogger(__name__)


async def detect_patterns(owner_id: int) -> list[dict]:
    """
    Анализирует память и возвращает список инсайтов.
    Каждый инсайт: {"type": str, "title": str, "detail": str, "action": str}
    Типы: "periodic_contact", "stale_negative", "sentiment_shift", "unfinished_topic"
    """
    insights: list[dict] = []
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        memories = await list_memories(session, owner)
        contacts = await list_contacts(
            session, owner, kinds=("user",), include_bots=False
        )

        # ---- Инсайт 1: периодические контакты ----
        contact_activity: dict[int, list[datetime]] = defaultdict(list)
        for m in memories:
            if m.contact_id is not None and m.created_at is not None:
                contact_activity[m.contact_id].append(m.created_at)

        for contact_id, dates in contact_activity.items():
            if len(dates) >= 3:
                wdays: dict[int, int] = defaultdict(int)
                for d in dates:
                    wdays[d.weekday()] += 1
                best_day = max(wdays, key=wdays.get)  # type: ignore[arg-type]
                if wdays[best_day] >= 3:  # 3+ совпадений дня недели
                    contact = await get_contact(session, owner, contact_id)
                    name = contact.display_name if contact else str(contact_id)
                    day_names = ["ПН", "ВТ", "СР", "ЧТ", "ПТ", "СБ", "ВС"]
                    insights.append(
                        {
                            "type": "periodic_contact",
                            "contact_id": contact_id,
                            "title": f"📅 Регулярный контакт: {name}",
                            "detail": f"Ты общаешься с {name} каждую {day_names[best_day]} ({wdays[best_day]} раз за период).",
                            "action": f"Поставить еженедельное напоминание на {day_names[best_day]}?",
                        }
                    )

        # ---- Инсайт 2: забытые негативные контакты ----
        now = datetime.now(timezone.utc)
        contact_last_neg: dict[int, tuple[datetime, str]] = {}
        for m in memories:
            if (
                m.sentiment in ("negative", "contradictory")
                and m.contact_id
                and m.created_at
            ):
                if (
                    m.contact_id not in contact_last_neg
                    or m.created_at > contact_last_neg[m.contact_id][0]
                ):
                    contact_last_neg[m.contact_id] = (m.created_at, m.fact)

        for contact_id, (last_date, fact) in contact_last_neg.items():
            days_since = (now - last_date).days
            if days_since > 14:
                contact = await get_contact(session, owner, contact_id)
                name = contact.display_name if contact else str(contact_id)
                insights.append(
                    {
                        "type": "stale_negative",
                        "contact_id": contact_id,
                        "title": f"⚠️ Давно без контакта: {name}",
                        "detail": f"Последний негативный факт {days_since} дн. назад: «{fact[:80]}». Может написать?",
                        "action": f"Открыть /threads и проверить переписку с {name}",
                    }
                )

        # ---- Инсайт 3: сдвиг настроения ----
        contact_sentiments: dict[int, list[tuple[datetime, str]]] = defaultdict(list)
        for m in memories:
            if m.contact_id and m.sentiment and m.created_at:
                contact_sentiments[m.contact_id].append((m.created_at, m.sentiment))

        for contact_id, sent_list in contact_sentiments.items():
            if len(sent_list) >= 5:
                sorted_list = sorted(sent_list, key=lambda x: x[0])
                mid = len(sorted_list) // 2
                old = [s[1] for s in sorted_list[:mid]]
                new = [s[1] for s in sorted_list[mid:]]
                old_neg = sum(
                    1 for s in old if s in ("negative", "contradictory")
                ) / len(old)
                new_neg = sum(
                    1 for s in new if s in ("negative", "contradictory")
                ) / len(new)
                if new_neg - old_neg > 0.3:  # ухудшение
                    contact = await get_contact(session, owner, contact_id)
                    name = contact.display_name if contact else str(contact_id)
                    insights.append(
                        {
                            "type": "sentiment_shift",
                            "contact_id": contact_id,
                            "title": f"📉 Ухудшение отношений: {name}",
                            "detail": f"Негатив вырос с {int(old_neg * 100)}% до {int(new_neg * 100)}%. Проверь что происходит.",
                            "action": f"Написать {name} или /chat {name}",
                        }
                    )
                elif old_neg - new_neg > 0.3:  # улучшение
                    contact = await get_contact(session, owner, contact_id)
                    name = contact.display_name if contact else str(contact_id)
                    insights.append(
                        {
                            "type": "sentiment_shift",
                            "contact_id": contact_id,
                            "title": f"📈 Улучшение отношений: {name}",
                            "detail": f"Негатив снизился с {int(old_neg * 100)}% до {int(new_neg * 100)}%. Отлично!",
                            "action": f"Закрепить успех — написать {name}",
                        }
                    )

    return insights


def insights_keyboard(insight: dict) -> InlineKeyboardMarkup | None:
    """Возвращает inline-клавиатуру для инсайта по его типу."""
    t = insight["type"]
    contact_id = insight.get("contact_id", 0)

    if t == "periodic_contact":
        return InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="📅 Поставить напоминание",
                        callback_data=f"pattern:remind:{contact_id}",
                    ),
                    InlineKeyboardButton(
                        text="🔕 Не сейчас", callback_data="pattern:dismiss"
                    ),
                ],
                [
                    InlineKeyboardButton(
                        text="📊 История контакта",
                        callback_data=f"pattern:history:{contact_id}",
                    ),
                ],
            ]
        )
    if t == "stale_negative":
        return InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="💬 Написать",
                        callback_data=f"pattern:write:{contact_id}",
                    ),
                    InlineKeyboardButton(
                        text="🔕 Не сейчас", callback_data="pattern:dismiss"
                    ),
                ],
            ]
        )
    if t == "sentiment_shift":
        return InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="💬 Написать",
                        callback_data=f"pattern:write:{contact_id}",
                    ),
                    InlineKeyboardButton(
                        text="📊 Анализ",
                        callback_data=f"pattern:history:{contact_id}",
                    ),
                ],
            ]
        )
    return None


def format_insights(
    insights: list[dict],
) -> tuple[str, list[InlineKeyboardMarkup | None]]:
    """Форматирует инсайты в HTML для отправки.

    Возвращает (текст, список клавиатур) — клавиатура для каждого инсайта.
    """
    if not insights:
        return (
            "🧠 Анализ паттернов: всё стабильно. Необычных паттернов не обнаружено.",
            [None],
        )
    lines: list[str] = ["<b>🧠 Инсайты из памяти:</b>", ""]
    keyboards: list[InlineKeyboardMarkup | None] = []
    for i, ins in enumerate(insights[:5]):
        lines.append(f"{i + 1}. {ins['title']}")
        lines.append(f"   {ins['detail']}")
        lines.append(f"   💡 {ins['action']}")
        lines.append("")
        keyboards.append(insights_keyboard(ins))
    return "\n".join(lines), keyboards


async def patterns_loop(owner_id: int) -> None:
    """Фоновый цикл: раз в 24 часа в 10:00 по часовому поясу владельца."""
    last_run_date: object = None
    while True:
        try:
            async with get_session() as session:
                owner = await get_or_create_user(session, owner_id)
                tz_name = owner.settings.timezone if owner.settings else "UTC"

            now = now_in_tz(tz_name)
            today = now.date()
            if now.hour == 10 and last_run_date != today:
                last_run_date = today
                insights = await detect_patterns(owner_id)
                text, keyboards = format_insights(insights)
                for ins, kb in zip(insights[:5], keyboards):
                    detail = (
                        f"<b>{ins['title']}</b>\n{ins['detail']}\n💡 {ins['action']}"
                    )
                    await notifier.notify(detail, reply_markup=kb)
                    await asyncio.sleep(0.5)
                await asyncio.sleep(600)  # не повторять в этот час
            await asyncio.sleep(600)  # проверка каждые 10 минут
        except Exception as e:
            logger.error(f"Patterns loop error: {e}")
            await asyncio.sleep(3600)
