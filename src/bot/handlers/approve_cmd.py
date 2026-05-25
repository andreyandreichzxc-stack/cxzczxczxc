"""Command: /approve <sender_id> <code> — approve a pending contact."""

from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from src.bot.filters import OwnerOnly
from src.core.security.pairing import pairing

router = Router(name="approve")
router.message.filter(OwnerOnly())


@router.message(Command("approve"))
async def cmd_approve(message: Message) -> None:
    args = message.text.split()
    if len(args) != 3:
        await message.answer("❌ Использование: /approve <id> <код>")
        return
    try:
        sender_id = int(args[1])
        code = args[2]
    except (ValueError, IndexError):
        await message.answer("❌ Неверный формат. Использование: /approve <id> <код>")
        return
    if await pairing.approve(sender_id, code):
        await message.answer(f"✅ Контакт {sender_id} одобрен.")
    else:
        await message.answer("❌ Неверный код или контакт не ожидает подтверждения.")


@router.message(Command("revoke"))
async def cmd_revoke(message: Message) -> None:
    args = message.text.split()
    if len(args) != 2:
        await message.answer("❌ Использование: /revoke <id>")
        return
    try:
        sender_id = int(args[1])
    except (ValueError, IndexError):
        await message.answer("❌ Неверный ID.")
        return
    await pairing.revoke(sender_id)
    await message.answer(f"✅ Доступ для {sender_id} отозван.")


@router.message(Command("pending"))
async def cmd_pending(message: Message) -> None:
    count = pairing.pending_count
    if count == 0:
        await message.answer("Нет ожидающих подтверждения.")
    else:
        await message.answer(f"⏳ Ожидают подтверждения: {count} контакт(ов).")
