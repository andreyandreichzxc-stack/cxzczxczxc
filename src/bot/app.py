import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramNetworkError, TelegramRetryAfter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.types import Message

from src.bot.handlers import (
    analyze_cmd,
    approve_cmd,
    ask_cmd,
    avito_cmd,
    catchup_cmd,
    chat_cmd,
    contact_cmd,
    digest_cmd,
    docs_cmd,
    draft_actions,
    explain_cmd,
    gates_cmd,
    health_cmd,
    help_cmd,
    humanize_cmd,
    inbox_cmd,
    inline_query,
    install_cmd,
    mode_cmd,
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
    sessions_cmd,
    settings as settings_handlers,
    skills_cmd,
    start,
    style_cmd,
    threads_cmd,
    timeline_cmd,
    today_cmd,
    todos,
    trajectory_cmd,
    wiki_cmd,
)
from src.bot.handlers.free_text_pipeline import confirm_router
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

    # ─── Онбординг-гард: фазовая блокировка команд ───
    @dp.message.outer_middleware()
    async def onboarding_guard_middleware(
        handler, message: Message, data: dict
    ) -> None:
        """Перенаправляет не-онбордингнутых пользователей на нужный шаг.

        Фазы:
          1 (нет сессии)     — только /start, /login, /cancel
          2 (нет LLM-ключа)  — плюс /keys, /settings
          3 (нет часового)   — всё разрешено, но подсказка /sync после ответа
          4 (готов)          — без ограничений
        """
        if not message.from_user:
            return  # channel posts, no user context

        tg_id = message.from_user.id
        if tg_id != settings.owner_telegram_id:
            return await handler(message, data)

        # Всегда пропускаем команды онбординга
        text = message.text or ""
        if text.startswith(("/start", "/login", "/cancel")):
            return await handler(message, data)

        # Если пользователь в любом FSM — не вмешиваемся
        state: FSMContext | None = data.get("state")
        if state is not None:
            current = await state.get_state()
            if current is not None:
                return await handler(message, data)

        from src.bot.filters import get_onboarding_phase

        phase = await get_onboarding_phase(tg_id)

        # Фаза 4 — всё настроено, пропускаем
        if phase == 4:
            return await handler(message, data)

        # Фаза 1 — нет сессии: только /start, /login, /cancel (уже пропущены выше)
        if phase == 1:
            await message.answer("Сначала сделай /login")
            return

        # Фаза 2 — нет LLM-ключа: разрешаем /keys и /settings
        if phase == 2:
            if text.startswith(("/keys", "/settings")):
                return await handler(message, data)
            await message.answer("Теперь добавь API-ключ для LLM. Жми /keys add.")
            return

        # Фаза 3 — нет часового пояса / синхронизации:
        # разрешаем всё, но после ответа показываем подсказку /sync
        await handler(message, data)
        try:
            await message.answer("💡 Хочешь чтобы я запомнил важное? Сделай /sync.")
        except Exception:
            pass
        return

    # Inline-режим — самый первый, чтобы ловить @botname до команд
    dp.include_router(inline_query.router)
    dp.include_router(approve_cmd.router)
    dp.include_router(ask_cmd.router)
    dp.include_router(gates_cmd.router)
    dp.include_router(health_cmd.router)
    dp.include_router(help_cmd.router)
    dp.include_router(docs_cmd.router)
    dp.include_router(inbox_cmd.router)
    dp.include_router(install_cmd.router)
    dp.include_router(start.router)
    dp.include_router(analyze_cmd.router)
    dp.include_router(contact_cmd.router)
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
    dp.include_router(timeline_cmd.router)
    dp.include_router(sessions_cmd.router)
    dp.include_router(explain_cmd.router)
    dp.include_router(humanize_cmd.router)
    dp.include_router(mode_cmd.router)
    dp.include_router(today_cmd.router)
    dp.include_router(skills_cmd.router)
    dp.include_router(trajectory_cmd.router)
    dp.include_router(wiki_cmd.router)
    dp.include_router(avito_cmd.router)
    dp.include_router(free_text_memory.router)
    dp.include_router(free_text_settings.router)
    dp.include_router(confirm_router)
    # ВАЖНО: free_text — самым последним, чтобы команды и FSM перехватили текст раньше
    dp.include_router(free_text.router)

    me = await bot.get_me()
    logger.info("Control bot started as @%s", me.username)

    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        await bot.session.close()
