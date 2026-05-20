# NOTE: Для некритических уведомлений используй notification_queue.enqueue()
# вместо notifier.notify(). Прямой вызов notifier.notify() — только для CRITICAL.
import logging
from typing import TYPE_CHECKING

from src.config import settings
from src.bot.tg_sender import send_with_retry


if TYPE_CHECKING:
    from aiogram import Bot
    from aiogram.types import InlineKeyboardMarkup


logger = logging.getLogger(__name__)


class Notifier:
    # шлёт сообщения владельцу через control bot — используется userbot-кодом

    def __init__(self) -> None:
        self._bot: "Bot | None" = None

    def attach(self, bot: "Bot") -> None:
        self._bot = bot

    async def notify(
        self,
        text: str,
        *,
        parse_mode: str | None = "HTML",
        reply_markup: "InlineKeyboardMarkup | None" = None,
    ) -> None:
        if self._bot is None:
            logger.warning("Notifier not attached, skipping: %s", text[:80])
            return
        try:
            await send_with_retry(
                self._bot.send_message,
                chat_id=settings.owner_telegram_id,
                text=text,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
            )
        except Exception as e:
            logger.error("Failed to notify owner after retries: %s", e)


notifier = Notifier()
