import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiohttp import ClientSession

from src.bot.handlers import (
    analyze_cmd,
    catchup_cmd,
    chat_cmd,
    digest_cmd,
    draft_actions,
    explain_cmd,
    free_text,
    free_text_memory,
    free_text_settings,
    login,
    memory_cmd,
    news_cmd,
    news_topics,
    profile_cmd,
    search,
    send,
    settings as settings_handlers,
    skills_cmd,
    start,
    style_cmd,
    threads_cmd,
    today_cmd,
    todos,
    trajectory_cmd,
)
from src.config import settings
from src.core.infra.notifier import notifier
from src.userbot.manager import UserbotManager


logger = logging.getLogger(__name__)


async def run_bot(userbot_manager: UserbotManager) -> None:
    session_kwargs = {}
    if settings.proxy_url:
        session_kwargs["proxy"] = settings.proxy_url
    session = ClientSession(**session_kwargs) if session_kwargs else None

    bot = Bot(
        token=settings.bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
        session=session,
    )
    notifier.attach(bot)

    dp = Dispatcher(storage=MemoryStorage())

    dp["userbot_manager"] = userbot_manager

    dp.include_router(start.router)
    dp.include_router(analyze_cmd.router)
    dp.include_router(profile_cmd.router)
    dp.include_router(login.router)
    dp.include_router(settings_handlers.router)
    dp.include_router(chat_cmd.router)
    dp.include_router(catchup_cmd.router)
    dp.include_router(send.router)
    dp.include_router(search.router)
    dp.include_router(todos.router)
    dp.include_router(digest_cmd.router)
    dp.include_router(style_cmd.router)
    dp.include_router(memory_cmd.router)
    dp.include_router(news_cmd.router)
    dp.include_router(draft_actions.router)
    dp.include_router(news_topics.router)
    dp.include_router(threads_cmd.router)
    dp.include_router(explain_cmd.router)
    dp.include_router(today_cmd.router)
    dp.include_router(skills_cmd.router)
    dp.include_router(trajectory_cmd.router)
    dp.include_router(free_text_memory.router)
    dp.include_router(free_text_settings.router)
    # ВАЖНО: free_text — самым последним, чтобы команды и FSM перехватили текст раньше
    dp.include_router(free_text.router)

    me = await bot.get_me()
    logger.info("Control bot started as @%s", me.username)

    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        await bot.session.close()
