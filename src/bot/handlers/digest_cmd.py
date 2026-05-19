import re
from datetime import datetime

from aiogram import Router
from aiogram.filters import Command, CommandObject
from aiogram.types import Message

from src.bot.filters import OwnerOnly
from src.core.digest import build_digest
from src.core.smart_digest import build_smart_digest, collect_recent_messages
from src.core.timeutil import tz_short
from src.db.repo import get_or_create_user
from src.db.session import get_session


router = Router(name="digest_cmd")
router.message.filter(OwnerOnly())


HM_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


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
    await message.answer(text)


@router.message(Command("smart_digest"))
async def cmd_smart_digest(message: Message) -> None:
    """Ручной запуск smart-дайджеста."""
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        interval = owner.settings.smart_digest_interval_min
        messages = await collect_recent_messages(session, owner, since_minutes=interval)
        text = build_smart_digest(messages, interval)
        owner.settings.smart_digest_last_sent = datetime.utcnow()
    await message.answer(text)
