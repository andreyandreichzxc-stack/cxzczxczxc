"""Централизованная отправка сообщений с retry/backoff и авто-санитацией HTML."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from aiogram.exceptions import TelegramNetworkError, TelegramRetryAfter

from src.core.infra.text_sanitizer import sanitize_html

logger = logging.getLogger(__name__)


async def send_with_retry(
    send_fn,  # async callable
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 2.0,
    **kwargs: Any,
) -> Any:
    """Вызвать send_fn(*args, **kwargs) с retry при FloodWaitError/RateLimit.

    Обрабатывает:
    - aiogram.exceptions.TelegramRetryAfter → ждать + retry
    - aiogram.exceptions.TelegramNetworkError → retry с backoff
    - telethon.errors.FloodWaitError → ждать указанное время + retry
    """
    for attempt in range(max_retries):
        try:
            return await send_fn(*args, **kwargs)
        except TelegramRetryAfter as e:
            delay = max(e.retry_after, base_delay * (2**attempt))
            logger.warning(
                "Telegram 429: waiting %.1fs (attempt %d/%d)",
                delay,
                attempt + 1,
                max_retries,
            )
            await asyncio.sleep(delay)
        except TelegramNetworkError as e:
            if attempt == max_retries - 1:
                logger.exception("Telegram network error, max retries reached")
                raise
            delay = base_delay * (2**attempt)
            logger.warning(
                "Telegram network error: %s, retrying in %.1fs (attempt %d/%d)",
                e,
                delay,
                attempt + 1,
                max_retries,
            )
            await asyncio.sleep(delay)
        except Exception as e:
            # Telethon FloodWaitError (optional dependency — string match)
            exc_name = type(e).__name__
            if "FloodWaitError" in exc_name:
                wait = getattr(e, "seconds", base_delay * (2**attempt))
                logger.warning(
                    "FloodWait %ds (attempt %d/%d)",
                    wait,
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(wait)
                continue
            # Неизвестная ошибка — не retry
            logger.exception("Unexpected error in send_with_retry: %s", exc_name)
            raise

    raise RuntimeError(f"Send failed after {max_retries} retries")


async def safe_send(
    bot: Any,
    chat_id: int | str,
    text: str,
    *,
    parse_mode: str | None = "HTML",
    reply_markup: Any = None,
    **kwargs: Any,
) -> Any:
    """Отправить сообщение с авто-санитацией HTML и retry/backoff.

    Если parse_mode="HTML", текст автоматически прогоняется через
    sanitize_html() перед отправкой.
    """
    safe_text = sanitize_html(text) if parse_mode == "HTML" else text
    return await send_with_retry(
        bot.send_message,
        chat_id=chat_id,
        text=safe_text,
        parse_mode=parse_mode,
        reply_markup=reply_markup,
        **kwargs,
    )
