"""Централизованная отправка сообщений с retry/backoff и авто-санитацией HTML."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.core.text_sanitizer import sanitize_html

logger = logging.getLogger(__name__)


async def send_with_retry(
    send_fn,  # async callable
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    **kwargs: Any,
) -> Any:
    """Вызвать send_fn(*args, **kwargs) с retry при FloodWaitError/RateLimit.

    Обрабатывает:
    - telethon.errors.FloodWaitError → ждать указанное время + retry
    - aiogram.exceptions.TelegramRetryAfter → ждать + retry
    - aiogram.exceptions.TelegramNetworkError → retry с backoff
    """
    for attempt in range(max_retries):
        try:
            return await send_fn(*args, **kwargs)
        except Exception as e:
            exc_name = type(e).__name__

            # Telethon FloodWait
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

            # aiogram RetryAfter
            if "TelegramRetryAfter" in exc_name:
                wait = getattr(e, "retry_after", base_delay * (2**attempt))
                logger.warning(
                    "RetryAfter %ds (attempt %d/%d)",
                    wait,
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(wait)
                continue

            # aiogram NetworkError
            if "TelegramNetworkError" in exc_name:
                wait = base_delay * (2**attempt)
                logger.warning(
                    "NetworkError, retry in %ds (attempt %d/%d)",
                    wait,
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(wait)
                continue

            # Неизвестная ошибка — не retry
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
