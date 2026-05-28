"""Command: /wiki — show Memory Wiki."""

from __future__ import annotations

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from src.bot.filters import OwnerOnly
from src.core.memory.memory_wiki import generate_memory_wiki, WIKI_DIR

router = Router()
router.message.filter(OwnerOnly())


@router.message(Command("wiki"))
async def cmd_wiki(message: Message) -> None:
    """Generate and show Memory Wiki."""
    await message.answer("📚 Генерирую Memory Wiki...")

    try:
        await generate_memory_wiki(message.from_user.id)
        index_path = WIKI_DIR / "index.md"
        if index_path.exists():
            content = index_path.read_text(encoding="utf-8")
            # Truncate to fit Telegram message
            if len(content) > 3800:
                content = content[:3800] + "\n\n... (обрезано)"
            await message.answer(content)
        else:
            await message.answer("❌ Не удалось создать wiki.")
    except Exception as e:
        await message.answer(f"❌ Ошибка генерации wiki: {e}")
