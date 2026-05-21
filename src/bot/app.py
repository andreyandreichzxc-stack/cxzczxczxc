import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramNetworkError, TelegramRetryAfter
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.client.session.aiohttp import AiohttpSession

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


def _retry_wrapper(send_fn):
    """Wrap bot.send_message with exponential backoff on 429 / network errors.

    This covers ALL callers (message.answer(), notifier, safe_send, etc.)
    with zero changes to handler code.
    """

    async def wrapper(chat_id, text, **kwargs):
        max_retries = 3
        base_delay = 2.0
        for attempt in range(max_retries):
            try:
                return await send_fn(chat_id, text, **kwargs)
            except TelegramRetryAfter as e:
                delay = max(e.retry_after, base_delay * (2**attempt))
                logger.warning(
                    "Telegram 429: waiting %.1fs (attempt %d/%d)",
                    delay,
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(delay)
            except TelegramNetworkError:
                if attempt == max_retries - 1:
                    logger.exception("Telegram network error, max retries reached")
                    raise
                delay = base_delay * (2**attempt)
                logger.warning(
                    "Telegram network error, retrying in %.1fs (attempt %d/%d)",
                    delay,
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(delay)
        raise RuntimeError(f"send_message failed after {max_retries} retries")

    return wrapper


async def run_bot(userbot_manager: UserbotManager) -> None:
    session = AiohttpSession(proxy=settings.proxy_url) if settings.proxy_url else None

    bot = Bot(
        token=settings.bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
        session=session,
    )
    notifier.attach(bot)

    # Patch bot.send_message so ALL outbound messages (message.answer, etc.)
    # automatically get retry with exponential backoff.
    bot.send_message = _retry_wrapper(bot.send_message)

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
