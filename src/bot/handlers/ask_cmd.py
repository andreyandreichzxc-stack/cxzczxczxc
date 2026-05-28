"""Команда /ask — проанализировать чат через LLM с произвольным вопросом.

Usage:
    /ask <имя>                — анализ последних 50 сообщений
    /ask <имя> <N>           — анализ последних N сообщений
    /ask <имя> <вопрос>      — анализ + вопрос
    /ask <имя> <N> <вопрос>  — N сообщений + вопрос
"""

import re
import logging

from aiogram import Router
from aiogram.filters import Command, CommandObject
from aiogram.types import Message

from src.bot.filters import OwnerOnly
from src.core.contacts.contact_resolver import resolve
from src.core.services.chat_actions import ask_chat_action
from src.db.repo import get_or_create_user
from src.db.session import get_session
from src.userbot.manager import UserbotManager

logger = logging.getLogger(__name__)

router = Router(name="ask_cmd")
router.message.filter(OwnerOnly())


def _parse_ask_args(text: str) -> tuple[str, int, str]:
    """Парсит аргументы /ask: (contact_name, limit, query)"""
    # Убираем /ask
    stripped = re.sub(r"^/ask\s*", "", text, count=1).strip()
    if not stripped:
        return "", 50, ""

    # Имя может быть в кавычках
    name = ""
    rest = stripped

    q_match = re.match(r'"([^"]+)"\s*(.*)', stripped)
    if q_match:
        name = q_match.group(1)
        rest = q_match.group(2).strip()
    else:
        # Просто первое слово — имя
        parts = stripped.split(None, 1)
        name = parts[0]
        rest = parts[1] if len(parts) > 1 else ""

    # Пробуем вытащить N из rest
    limit = 50
    query = rest

    if rest:
        num_match = re.match(r"^(\d+)\s*(.*)", rest)
        if num_match:
            limit = int(num_match.group(1))
            limit = max(1, min(limit, 500))  # clamp 1-500
            query = num_match.group(2).strip()

    return name, limit, query


@router.message(Command("ask"))
async def ask_cmd(
    message: Message,
    command: CommandObject,
    userbot_manager: UserbotManager,
) -> None:
    """Обработчик /ask — анализ чата через LLM."""
    text = command.args or ""
    if not text:
        await message.answer(
            "📋 <b>Как пользоваться /ask:</b>\n\n"
            "<code>/ask Имя</code> — анализ последних 50 сообщений\n"
            "<code>/ask Имя 100</code> — последних 100\n"
            "<code>/ask Имя что думаешь?</code> — с вопросом\n"
            "<code>/ask Имя 100 что там?</code> — N + вопрос\n\n"
            'Имена с пробелами — в кавычках: <code>/ask "Иван Иванов"</code>'
        )
        return

    name, limit, query = _parse_ask_args(text)
    if not name:
        await message.answer("❌ Укажи имя контакта или чата.")
        return

    client = userbot_manager.get_client(message.from_user.id)
    if client is None:
        await message.answer("❌ Сначала /login.")
        return

    # Статус
    status_msg = await message.answer(f"🔍 Ищу чат «{name}»…")

    # Решаем имя → peer_id через fuzzy matching
    try:
        async with get_session() as session:
            owner = await get_or_create_user(session, message.from_user.id)
            candidates = await resolve(client, owner, name)
    except Exception as e:
        logger.error("resolve error for %s: %s", name, e)
        await status_msg.edit_text(f"❌ Ошибка поиска чата: {e}")
        return

    if not candidates:
        await status_msg.edit_text(
            f"❌ Чат «{name}» не найден.\n"
            "Проверь имя или попробуй синхронизировать контакты: /sync"
        )
        return

    peer_id = candidates[0].peer_id
    display_name = candidates[0].display_name

    # Анализируем
    limit_str = f" (последние {limit} сообщ.)"
    query_str = f", вопрос: {query[:60]}…" if query else ""
    await status_msg.edit_text(
        f"🤖 Анализирую чат «{display_name}»{limit_str}{query_str}…"
    )

    try:
        action_result = await ask_chat_action(
            telegram_id=message.from_user.id,
            peer_id=peer_id,
            userbot_manager=userbot_manager,
            user_query=query,
            limit=limit,
        )
    except Exception as e:
        logger.error("ask_chat_action error for %s: %s", display_name, e)
        await status_msg.edit_text(f"❌ Ошибка анализа: {e}")
        return

    if action_result is None:
        await status_msg.edit_text(
            f"❌ Не удалось загрузить сообщения из чата «{display_name}».\n"
            "Возможно, нет сообщений или нет доступа."
        )
        return

    # Отправляем результат
    await status_msg.edit_text(
        action_result.html,
        reply_markup=action_result.markup,
    )
