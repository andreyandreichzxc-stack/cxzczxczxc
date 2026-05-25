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
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    from src.db.models import User
    from src.db.session import get_session

    async with get_session() as session:
        stmt = (
            select(User)
            .where(User.telegram_id == tg_id)
            .options(
                selectinload(User.session),
                selectinload(User.key_slots),
                selectinload(User.settings),
            )
        )
        owner = (await session.execute(stmt)).unique().scalar_one_or_none()
        if owner is None:
            return False
        has_session = owner.session is not None
        has_llm_key = len(owner.key_slots) > 0
        has_tz = owner.settings.timezone not in (None, "", "UTC", "Etc/UTC")
    return has_session and has_llm_key and has_tz


async def get_onboarding_phase(tg_id: int) -> int:
    """Возвращает фазу онбординга (1–4).

    Фазы:
      1 — нет Telegram-сессии (только /start, /login, /cancel)
      2 — нет LLM-ключа (плюс /keys add)
      3 — нет часового пояса / синхронизации (всё разрешено, но с подсказкой /sync)
      4 — онбординг завершён
    """
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    from src.db.models import User
    from src.db.session import get_session

    async with get_session() as session:
        stmt = (
            select(User)
            .where(User.telegram_id == tg_id)
            .options(
                selectinload(User.session),
                selectinload(User.key_slots),
                selectinload(User.settings),
            )
        )
        owner = (await session.execute(stmt)).unique().scalar_one_or_none()
        if owner is None:
            return 1  # незнакомец — фаза 1

        has_session = owner.session is not None
        has_llm_key = len(owner.key_slots) > 0
        has_tz = owner.settings.timezone not in (None, "", "UTC", "Etc/UTC")

    if not has_session:
        return 1
    if not has_llm_key:
        return 2
    if not has_tz:
        return 3
    return 4
