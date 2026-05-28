"""Команда /help — справочник по командам бота с группировкой по категориям."""

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from src.bot.filters import OwnerOnly

router = Router(name="help")
router.message.filter(OwnerOnly())

HELP_TEXT = """\
<b>📋 Команды бота</b>

<b>👤 Люди</b>
/contact Имя — что я знаю о человеке
/timeline тема — хронология обсуждений
/send Имя текст — написать человеку
/style Имя — мой стиль общения с человеком

<b>🧠 Память и задачи</b>
/todos — список обещаний
/remember факт — запомнить факт
/forget запрос — забыть факты

<b>🔍 Поиск</b>
/search текст — поиск в чатах
/index Имя — проиндексировать чат

<b>📰 Новости и дайджест</b>
/news тема — дайджест каналов
/news_channels — источники новостей
/digest — утренний дайджест

<b>🔄 Аккаунт</b>
/login — авторизоваться
/logout — выйти
/sync — обновить контакты

<b>🔑 Ключи и API</b>
/keys — слоты ключей
/keys add — добавить ключ
/keys import — импорт списком
/gates — проверка зависимостей
/docs endpoints — инструкция по кастомным API

<b>⚙️ Настройки</b>
/settings — всё через меню
/humanize текст — анализ AI-шаблонности

<b>✏️ Обычный язык</b>
Просто напиши или скажи голосом — я пойму и сделаю.
"""


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    """Показать справочник команд, сгруппированный по категориям."""
    await message.answer(HELP_TEXT)
