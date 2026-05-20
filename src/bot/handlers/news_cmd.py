"""/news <тема> и /news_channels — управление новостными каналами."""
import re

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import CallbackQuery, InlineKeyboardButton, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder

from src.bot.filters import OwnerOnly
from src.core.scheduling.news import build_news_digest
from src.db.repo import (
    get_or_create_user,
    list_contacts,
    set_news_source,
)
from src.db.session import get_session
from src.userbot.manager import UserbotManager


router = Router(name="news_cmd")
router.message.filter(OwnerOnly())
router.callback_query.filter(OwnerOnly())


HOURS_RE = re.compile(r"--hours\s*=?\s*(\d+)")


@router.message(Command("news"))
async def cmd_news(message: Message, command: CommandObject, userbot_manager: UserbotManager) -> None:
    client = userbot_manager.get_client(message.from_user.id)
    if client is None:
        await message.answer("Сначала /login.")
        return

    raw = (command.args or "").strip()
    if not raw:
        await message.answer(
            "Использование: <code>/news тема [--hours=24]</code>\n"
            "Например: <code>/news AI и регулирование --hours=48</code>"
        )
        return

    # парсим --hours
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        default_hours = owner.settings.news_window_hours

    hours = default_hours
    m = HOURS_RE.search(raw)
    if m:
        hours = max(1, min(168, int(m.group(1))))
        raw = HOURS_RE.sub("", raw).strip()

    topic = raw
    if not topic:
        await message.answer("Укажи тему после команды.")
        return

    await message.answer(f"📰 Готовлю дайджест по «<i>{topic}</i>» за последние {hours}ч…")
    text = await build_news_digest(client, message.from_user.id, topic, hours=hours)
    await message.answer(text, disable_web_page_preview=True)


@router.message(Command("news_channels"))
async def cmd_news_channels(message: Message, userbot_manager: UserbotManager) -> None:
    client = userbot_manager.get_client(message.from_user.id)
    if client is None:
        await message.answer("Сначала /login.")
        return

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        channels = await list_contacts(session, owner, kinds=("channel",))

    if not channels:
        await message.answer("Каналов в БД нет. Запусти /sync.")
        return

    marked = sum(1 for c in channels if c.is_news_source)
    await message.answer(
        f"📰 <b>Каналы для /news</b>\n\n"
        f"Всего каналов: {len(channels)}\n"
        f"Помечено как источники: <b>{marked}</b>\n\n"
        "Тапни по каналу, чтобы переключить статус. Если ни один не помечен — /news берёт все."
    )

    # выводим пачками по 25 кнопок (Telegram-лимит ~100 кнопок в одном сообщении, но 1 строка ≤ 8)
    chunk = 20
    for i in range(0, len(channels), chunk):
        kb = InlineKeyboardBuilder()
        for c in channels[i:i + chunk]:
            mark = "✅" if c.is_news_source else "▫"
            label = f"{mark} {c.display_name[:40]}"
            kb.row(InlineKeyboardButton(text=label, callback_data=f"news:tog:{c.peer_id}"))
        await message.answer(f"Список ({i + 1}–{i + len(channels[i:i+chunk])}):", reply_markup=kb.as_markup())


@router.callback_query(F.data.startswith("news:tog:"))
async def cb_toggle(callback: CallbackQuery) -> None:
    peer_id = int(callback.data.split(":")[2])
    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        # инвертируем
        contacts = await list_contacts(session, owner, kinds=("channel",))
        target = next((c for c in contacts if c.peer_id == peer_id), None)
        if target is None:
            await callback.answer("Канал не найден", show_alert=True)
            return
        new_value = not target.is_news_source
        await set_news_source(session, owner, peer_id, new_value)

    label_text = ("✅ " if new_value else "▫ ") + (target.display_name[:40])
    # обновляем нужную кнопку: проще — отвечаем тостом + редактируем reply_markup
    if callback.message and callback.message.reply_markup:
        new_kb = []
        for row in callback.message.reply_markup.inline_keyboard:
            new_row = []
            for btn in row:
                if btn.callback_data == callback.data:
                    new_row.append(InlineKeyboardButton(text=label_text, callback_data=btn.callback_data))
                else:
                    new_row.append(btn)
            new_kb.append(new_row)
        from aiogram.types import InlineKeyboardMarkup
        await callback.message.edit_reply_markup(reply_markup=InlineKeyboardMarkup(inline_keyboard=new_kb))
    await callback.answer("Включено" if new_value else "Выключено")
