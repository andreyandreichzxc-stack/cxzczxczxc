from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from src.bot.filters import OwnerOnly
from src.db.repo import get_or_create_user
from src.db.session import get_session


router = Router(name="start")
router.message.filter(OwnerOnly())


WELCOME = (
    "👋 Я твой AI-ассистент для Telegram.\n\n"
    "<b>Аккаунт</b>\n"
    "/login — подключить Telegram-аккаунт (api_id, api_hash, телефон, код, 2FA)\n"
    "/logout — удалить сохранённую сессию\n"
    "/sync — обновить список контактов из диалогов\n\n"
    "<b>Настройки</b>\n"
    "/settings — авто-ответ, выбор LLM, API-ключи\n\n"
    "<b>Работа с чатами</b>\n"
    "/chat &lt;имя&gt; — саммари, задачи, черновик ответа, «где остановились»\n"
    "/catchup &lt;имя&gt; — где мы остановились + черновик ответа\n"
    "/search &lt;текст&gt; — поиск по проиндексированным сообщениям\n"
    "/index &lt;имя&gt; — проиндексировать чат для семантического поиска\n"
    "/send &lt;инструкция&gt; — «скажи Оле, что созвон в 8» (с подтверждением)\n\n"
    "<b>Новости</b>\n"
    "/news &lt;тема&gt; [--hours=24] — дайджест из подписанных каналов\n"
    "/news_channels — отметить каналы-источники\n"
    "/news_topics — темы для утренних авто-новостей\n\n"
    "<b>Память и фичи</b>\n"
    "/todos — открытые обещания (мои и мне)\n"
    "/digest [now|on|off|at HH:MM] — утренний дайджест\n"
    "/style &lt;имя&gt; — пересчитать профиль моего стиля общения с этим контактом\n\n"
    "/help — эта подсказка"
)


@router.message(Command("start", "help"))
async def cmd_start(message: Message) -> None:
    async with get_session() as session:
        await get_or_create_user(session, message.from_user.id)
    await message.answer(WELCOME)
