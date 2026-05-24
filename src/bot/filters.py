from aiogram.filters import BaseFilter
from aiogram.types import CallbackQuery, Message

from src.config import settings


class OwnerOnly(BaseFilter):
    """Допускает только владельца, указанного в OWNER_TELEGRAM_ID."""

    async def __call__(self, event: Message | CallbackQuery) -> bool:
        user = event.from_user
        return user is not None and user.id == settings.owner_telegram_id


async def is_onboarded(tg_id: int) -> bool:
    """Проверяет, прошёл ли пользователь полный онбординг.

    Критерии:
      - есть активная Telegram-сессия
      - есть хотя бы один LLM-ключ (LlmKeySlot)
      - часовой пояс отличается от UTC (или "Europe/Moscow" и т.п.)
    """
    from src.db.repo import get_or_create_user
    from src.db.session import get_session

    async with get_session() as session:
        owner = await get_or_create_user(session, tg_id)
        has_session = owner.session is not None
        has_llm_key = len(owner.key_slots) > 0
        has_tz = owner.settings.timezone not in (None, "", "UTC", "Etc/UTC")
    return has_session and has_llm_key and has_tz
