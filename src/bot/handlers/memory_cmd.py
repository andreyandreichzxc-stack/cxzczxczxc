import logging

from aiogram import Router
from aiogram.filters import Command, CommandObject
from aiogram.types import Message

from src.bot.filters import OwnerOnly
from src.core.contact_resolver import resolve
from src.db.repo import (
    add_memory,
    delete_memory,
    get_or_create_user,
    list_memories,
    search_memories,
)
from src.db.session import get_session
from src.userbot.manager import UserbotManager


logger = logging.getLogger(__name__)
router = Router(name="memory_cmd")
router.message.filter(OwnerOnly())


@router.message(Command("memory"))
async def cmd_memory(message: Message, userbot_manager: UserbotManager) -> None:
    """Показать память — всё или про конкретный контакт."""
    args = (message.text or "").replace("/memory", "").strip()

    contact_id = None
    label = ""
    if args:
        contact_name = args.strip()
        async with get_session() as session:
            owner = await get_or_create_user(session, message.from_user.id)
        client = (
            userbot_manager.get_client(message.from_user.id)
            if userbot_manager
            else None
        )
        if client is not None:
            candidates = await resolve(client, owner, contact_name)
            if candidates:
                contact_id = candidates[0].peer_id
                label = f" — {candidates[0].label()}"

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        items = await list_memories(session, owner, contact_id=contact_id)

    if not items:
        await message.answer("Память пуста.")
        return

    lines = []
    for m in items:
        sid = f"#{m.id}"
        sent = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(
            m.sentiment or "", ""
        )
        lines.append(f"• {sent} {sid} {m.fact}")
    body = "\n".join(lines)
    await message.answer(f"🧠 <b>Память{label}</b>\n\n{body}")


@router.message(Command("remember"))
async def cmd_remember(
    message: Message, command: CommandObject, userbot_manager: UserbotManager
) -> None:
    """Вручную сохранить факт. /remember Настя злится из-за дедлайна"""
    args = (command.args or "").strip()
    if not args:
        await message.answer(
            "Использование: <code>/remember [контакт] факт</code>\nПример: <code>/remember Настя злится</code>"
        )
        return

    # пробуем отделить имя контакта от факта
    contact_name = None
    fact = args
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
    client = (
        userbot_manager.get_client(message.from_user.id) if userbot_manager else None
    )
    if client is not None:
        candidates = await resolve(client, owner, args)
        if candidates and candidates[0].score >= 70:
            contact_name = candidates[0].label()
            # пытаемся отделить: берём первое слово как имя
            words = args.split(None, 1)
            if len(words) > 1:
                fact = words[1]

    contact_id = None
    if contact_name:
        candidates = await resolve(client, owner, contact_name)
        if candidates:
            contact_id = candidates[0].peer_id

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        mem = await add_memory(
            session, owner, fact=fact, contact_id=contact_id, source="user"
        )

    await message.answer(f"🧠 Запомнил: <i>{fact}</i>")


@router.message(Command("forget"))
async def cmd_forget(
    message: Message, command: CommandObject, userbot_manager: UserbotManager
) -> None:
    """Удалить факты по подстроке. /forget злится"""
    args = (command.args or "").strip()
    if not args:
        await message.answer("Использование: <code>/forget часть текста</code>")
        return

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        found = await search_memories(session, owner, args)

    if not found:
        await message.answer("Ничего не нашёл.")
        return

    async with get_session() as session:
        for m in found:
            await delete_memory(session, owner, m.id)

    names = ", ".join(
        f"«{m.fact[:50]}…»" if len(m.fact) > 50 else f"«{m.fact}»" for m in found
    )
    await message.answer(f"🗑 Забыл: {names}")
