"""/today — главный пульт управления ответами. /radar — быстрый срез."""

import logging
from datetime import datetime, timedelta, timezone

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder

from src.bot.filters import OwnerOnly
from src.db.session import get_session
from src.db.repo import (
    get_or_create_user,
    get_contact,
    get_self_profile,
    list_open_commitments,
    list_memories,
    get_memory_stats,
    list_memory_candidates,
)
from src.core.reply_radar import collect_reply_radar, format_radar, RadarItem
from src.core.memory_health import calculate_health_score, format_health_compact
from src.core.memory_fuel import get_fuel_stats, format_fuel_line

logger = logging.getLogger(__name__)
router = Router(name="today_cmd")
router.message.filter(OwnerOnly())
router.callback_query.filter(OwnerOnly())


def _radar_keyboard(item: RadarItem) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.row(
        InlineKeyboardButton(
            text="💬 Открыть", callback_data=f"thread:open:{item.peer_id}"
        ),
        InlineKeyboardButton(
            text="✍️ Черновик", callback_data=f"thread:reply:{item.peer_id}"
        ),
    )
    kb.row(
        InlineKeyboardButton(
            text="🤔 Почему", callback_data=f"radar:why:{item.peer_id}"
        ),
        InlineKeyboardButton(
            text="⏰ Позже", callback_data=f"radar:snooze:{item.peer_id}"
        ),
    )
    return kb.as_markup()


@router.message(Command("today"))
async def cmd_today(message: Message):
    """Главный пульт: радар + обязательства + память + конфликты."""
    telegram_id = message.from_user.id
    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)

        # Reply Radar
        radar = await collect_reply_radar(telegram_id, limit=5)

        # Commitments
        commits = await list_open_commitments(session, owner)
        now = datetime.now(timezone.utc)
        overdue = [c for c in commits if c.deadline_at and c.deadline_at < now][:3]
        today = [
            c
            for c in commits
            if c.deadline_at
            and c.deadline_at >= now
            and c.deadline_at <= now.replace(hour=23, minute=59, second=59)
        ][:3]

        # Memory inbox
        candidates = await list_memory_candidates(session, owner)

        # Health
        health = await calculate_health_score(telegram_id)

        # Streak
        streak = await _daily_reply_streak(telegram_id)

    lines = ["<b>📡 Пульт управления</b>", ""]

    # Radar
    if radar:
        lines.append(f"<b>📡 Reply Radar ({len(radar)}):</b>")
        for item in radar:
            risk = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                item.risk_level, "⚪"
            )
            window = f" 🕐 {item.reply_window}" if item.reply_window else ""
            lines.append(
                f"{risk} <b>{item.contact_name}</b> — {item.waiting_hours:.0f}ч [{item.score}]{window}"
            )
            if item.latest_snippet:
                lines.append(f"   «{item.latest_snippet[:80]}»")
            lines.append(f"   <i>{item.reason}</i>")
        lines.append("")
    else:
        lines.append("✅ <b>Нет срочных ответов.</b>")

    # Commitments
    if overdue:
        lines.append(f"🔴 <b>Просрочено ({len(overdue)}):</b>")
        for c in overdue:
            lines.append(f"• {c.text[:80]}")
    if today:
        lines.append(f"🟡 <b>Сегодня ({len(today)}):</b>")
        for c in today:
            lines.append(f"• {c.text[:80]}")
    if overdue or today:
        lines.append("")

    # Memory inbox
    if candidates:
        lines.append(
            f"📥 <b>Memory Inbox:</b> {len(candidates)} неподтверждённых фактов. /memory --inbox"
        )
        lines.append("")

    # Health + Streak
    health_line = format_health_compact(health)
    lines.append(health_line)
    if streak:
        lines.append(streak)
    lines.append("")
    lines.append(
        "<i>/radar — только ответы | /warnings — конфликты | /habits — привычки</i>"
    )

    text = "\n".join(lines)

    # Кнопки для top-3 радара
    if radar:
        for item in radar[:3]:
            await message.answer(
                f"{'🔴' if item.risk_level == 'high' else '🟡' if item.risk_level == 'medium' else '🟢'} "
                f"<b>{item.contact_name}</b> — {item.waiting_hours:.0f}ч ({item.unread_count} непроч.)",
                reply_markup=_radar_keyboard(item),
            )

    await message.answer(text)


@router.message(Command("radar"))
async def cmd_radar(message: Message):
    """Только Reply Radar — быстрый срез."""
    radar = await collect_reply_radar(message.from_user.id, limit=5)
    text = format_radar(radar)
    await message.answer(text)
    if radar:
        for item in radar[:3]:
            await message.answer(
                f"<b>{item.contact_name}</b> — {item.score} баллов",
                reply_markup=_radar_keyboard(item),
            )


# Callback: radar:why
@router.callback_query(F.data.startswith("radar:why:"))
async def cb_radar_why(callback: CallbackQuery):
    peer_id = int(callback.data.split(":")[2])
    telegram_id = callback.from_user.id
    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)
        contact = await get_contact(session, owner, peer_id)
        name = contact.display_name if contact else str(peer_id)

    radar = await collect_reply_radar(telegram_id, limit=10)
    item = next((r for r in radar if r.peer_id == peer_id), None)
    if not item:
        await callback.answer("Контакт не в радаре")
        return

    lines = [
        f"<b>🤔 Почему {name} в радаре?</b>",
        f"Баллы: <b>{item.score}</b>",
        f"Риск: <b>{item.risk_level}</b>",
        f"Ждёт: {item.waiting_hours:.0f}ч",
        f"Непрочитано: {item.unread_count}",
        f"Причина: {item.reason}",
    ]
    if item.latest_snippet:
        lines.append(f"Последнее: «{item.latest_snippet}»")
    if item.memory_hints:
        lines.append("Факты памяти:")
        for h in item.memory_hints:
            lines.append(f"• {h}")
    if item.reply_window:
        lines.append(f"Лучшее время ответа: {item.reply_window}")
    await callback.message.answer("\n".join(lines))
    await callback.answer()


# Callback: radar:snooze
@router.callback_query(F.data.startswith("radar:snooze:"))
async def cb_radar_snooze(callback: CallbackQuery):
    peer_id = int(callback.data.split(":")[2])
    async with get_session() as session:
        from sqlalchemy import select, update
        from src.db.models import ConversationState

        res = await session.execute(
            select(ConversationState).where(
                ConversationState.user_id == callback.from_user.id,
                ConversationState.peer_id == peer_id,
            )
        )
        conv = res.scalar_one_or_none()
        if conv:
            conv.radar_snoozed_until = datetime.now(timezone.utc) + timedelta(hours=24)
            await session.commit()
    await callback.answer("Отложено на 24ч")
    if callback.message:
        await callback.message.edit_text(
            callback.message.text + "\n\n⏰ Отложено на 24ч"
        )


async def _daily_reply_streak(telegram_id: int) -> str:
    """Считает сколько дней подряд владелец отвечал в течение суток."""
    from src.db.models import ConversationState
    from sqlalchemy import select, func

    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)
        res = await session.execute(
            select(ConversationState)
            .where(
                ConversationState.user_id == owner.id,
                ConversationState.last_outgoing_at.is_not(None),
                ConversationState.last_incoming_at.is_not(None),
            )
            .order_by(ConversationState.last_outgoing_at.desc())
            .limit(100)
        )
        convos = list(res.scalars().all())

    if not convos:
        return ""

    now = datetime.now(timezone.utc)
    streak = 0
    for i in range(14):  # макс 14 дней назад
        day = (now - timedelta(days=i)).date()
        replied_that_day = False
        for c in convos:
            if c.last_incoming_at and c.last_outgoing_at:
                # Ответил в течение 24ч после входящего?
                if (
                    c.last_outgoing_at.date() == day
                    and (c.last_outgoing_at - c.last_incoming_at).total_seconds()
                    < 86400
                ):
                    replied_that_day = True
                    break
        if replied_that_day:
            streak += 1
        else:
            break

    if streak >= 2:
        return (
            f"\n🔥 <b>Daily streak:</b> отвечаешь в течение суток {streak} дней подряд!"
        )
    return ""
