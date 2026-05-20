from datetime import datetime, timezone

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from src.bot.filters import OwnerOnly
from src.core.digest import build_digest
from src.core.smart_digest import build_smart_digest, collect_recent_messages
from src.core.timeutil import HM_RE, tz_short
from src.db.repo import add_memory, get_or_create_user, list_contacts
from src.db.session import get_session


router = Router(name="digest_cmd")
router.message.filter(OwnerOnly())
router.callback_query.filter(OwnerOnly())


@router.message(Command("digest"))
async def cmd_digest(message: Message, command: CommandObject) -> None:
    arg = (command.args or "").strip().lower()

    if not arg or arg == "now":
        text = await build_digest(message.from_user.id)
        await message.answer(text)
        return

    if arg in {"on", "enable", "вкл"}:
        async with get_session() as session:
            owner = await get_or_create_user(session, message.from_user.id)
            owner.settings.digest_enabled = True
            tz = owner.settings.timezone
            digest_time = owner.settings.digest_time
        await message.answer(
            f"☀ Дайджест включён. Время: {digest_time} · {tz_short(tz)}.\n"
            "Изменить: /digest at HH:MM или /settings → Дайджест."
        )
        return

    if arg in {"off", "disable", "выкл"}:
        async with get_session() as session:
            owner = await get_or_create_user(session, message.from_user.id)
            owner.settings.digest_enabled = False
        await message.answer("Дайджест выключен.")
        return

    if arg.startswith("at "):
        hm = arg[3:].strip()
        if not HM_RE.match(hm):
            await message.answer(
                "Формат: <code>/digest at HH:MM</code> (в твоём TZ, напр. <code>06:30</code>)"
            )
            return
        async with get_session() as session:
            owner = await get_or_create_user(session, message.from_user.id)
            owner.settings.digest_time = hm
            owner.settings.digest_enabled = True
            tz = owner.settings.timezone
        await message.answer(f"☀ Дайджест будет в {hm} ежедневно · {tz_short(tz)}.")
        return

    await message.answer(
        "Использование:\n"
        "<code>/digest</code> — собрать сейчас\n"
        "<code>/digest on</code> | <code>off</code>\n"
        "<code>/digest at HH:MM</code> (в твоём часовом поясе)"
    )


@router.message(Command("briefing"))
async def cmd_briefing(message: Message) -> None:
    """Ручной запрос брифинга."""
    from src.core.proactive_briefing import collect_briefing_data, format_briefing

    data = await collect_briefing_data(message.from_user.id)
    text = format_briefing(data, "Брифинг")
    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="📬 Треды", callback_data="thread:refresh"),
                InlineKeyboardButton(text="📋 Задачи", callback_data="nav:todos"),
            ],
            [
                InlineKeyboardButton(
                    text="🔄 Обновить", callback_data="briefing:refresh"
                ),
            ],
        ]
    )
    await message.answer(text, reply_markup=kb)


@router.callback_query(F.data == "briefing:refresh")
async def cb_briefing_refresh(callback: CallbackQuery) -> None:
    """Обновить брифинг."""
    from src.core.proactive_briefing import collect_briefing_data, format_briefing

    data = await collect_briefing_data(callback.from_user.id)
    text = format_briefing(data, "Брифинг")
    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="📬 Треды", callback_data="thread:refresh"),
                InlineKeyboardButton(text="📋 Задачи", callback_data="nav:todos"),
            ],
            [
                InlineKeyboardButton(
                    text="🔄 Обновить", callback_data="briefing:refresh"
                ),
            ],
        ]
    )
    if callback.message:
        await callback.message.edit_text(text, reply_markup=kb)
    await callback.answer("Обновлено")


@router.message(Command("smart_digest"))
async def cmd_smart_digest(message: Message) -> None:
    """Ручной запуск smart-дайджеста."""
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        interval = owner.settings.smart_digest_interval_min
        messages = await collect_recent_messages(session, owner, since_minutes=interval)
        text = build_smart_digest(messages, interval)
        owner.settings.smart_digest_last_sent = datetime.now(timezone.utc).replace(
            tzinfo=None
        )
    await message.answer(text)


@router.message(Command("weekly"))
async def cmd_weekly(message: Message) -> None:
    """Ручной запуск недельного саммари."""
    from src.core.weekly_summarizer import summarize_contact_week
    from src.llm.router import build_provider

    await message.answer("📊 Запускаю недельное саммари...")
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        provider = await build_provider(session, owner)
        if not provider:
            await message.answer("❌ Нет LLM провайдера.")
            return

        import json as _json

        contacts = await list_contacts(
            session, owner, kinds=("user",), include_bots=False
        )
        monitored = (
            _json.loads(owner.settings.monitored_folders)
            if owner.settings.monitored_folders
            else []
        )
        if monitored:
            contacts = [
                c
                for c in contacts
                if any(
                    f.strip() in monitored for f in (c.folder_names or "").split(",")
                )
            ]

        total = 0
        for c in contacts[:10]:
            facts = await summarize_contact_week(provider, message.from_user.id, c)
            for f in facts:
                await add_memory(
                    session,
                    owner,
                    fact=f.get("fact", ""),
                    contact_id=c.peer_id,
                    sentiment=f.get("sentiment"),
                    source="weekly",
                )
                total += 1
        await message.answer(
            f"✅ Готово! {total} фактов сохранено в память "
            f"из {len(contacts[:10])} контактов."
        )
