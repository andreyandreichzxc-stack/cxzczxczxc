"""Session repository — User, UserSettings, TelegramSession, SelfProfile, AdaptivePersona."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.db.models import (
    AdaptivePersona,
    SelfProfile,
    TelegramSession,
    User,
    UserSettings,
)
from src.crypto import decrypt, encrypt

logger = logging.getLogger(__name__)

# Фоновые таски (digest, news, reminders, auto_sync) одновременно тыкаются в
# get_or_create_user на пустой БД → UNIQUE race. Сериализуем процесс-локом.
# Используем per-user/per-telegram-id локи вместо глобального, чтобы разные
# пользователи не блокировали друг друга.
_user_locks: dict[int, asyncio.Lock] = {}
# Счётчик вызовов для периодической синхронной чистки (каждые 1000 вызовов)
_lock_cleanup_counter: int = 0


def _get_user_lock(user_id: int) -> asyncio.Lock:
    """Возвращает per-user Lock, создаёт при первом обращении.

    Периодически (каждые 1000 вызовов) чистит неиспользуемые локи.
    Безопасно без доп. синхронизации: функция синхронная, без await —
    переключение контекста asyncio невозможно внутри неё, поэтому
    итерация + del атомарны относительно других вызовов.
    """
    global _lock_cleanup_counter
    if user_id not in _user_locks:
        _user_locks[user_id] = asyncio.Lock()
    _lock_cleanup_counter += 1
    if _lock_cleanup_counter % 1000 == 0:
        for k in list(_user_locks.keys()):
            if not _user_locks[k].locked():
                del _user_locks[k]
    return _user_locks[user_id]


async def get_or_create_user(
    session: AsyncSession, telegram_id: int, *, use_cache: bool = True
) -> User:
    if use_cache:
        from src.core.context_cache import get as cache_get

        cached = await cache_get(f"user:{telegram_id}")
        if cached is not None:
            # cached is user.id (int) — retrieve fresh session-attached object
            user = await session.get(
                User, cached, options=[selectinload(User.key_slots)]
            )
            if user is not None:
                return user
    lock = _get_user_lock(
        -telegram_id
    )  # отрицательный, чтобы не пересекаться с user.id
    async with lock:
        result = await session.execute(
            select(User)
            .where(User.telegram_id == telegram_id)
            .options(selectinload(User.key_slots))
        )
        user = result.scalar_one_or_none()
        if user is None:
            user = User(telegram_id=telegram_id, settings=UserSettings())
            session.add(user)
            await session.flush()
        elif user.settings is None:
            user.settings = UserSettings(user_id=user.id)
            session.add(user.settings)
            await session.flush()
    if use_cache:
        from src.core.context_cache import put as cache_put

        # Cache only the user.id (int) — not the ORM object — to avoid
        # returning a detached instance after the session closes.
        await cache_put(f"user:{telegram_id}", user.id, ttl=30)
    return user


async def save_telegram_session(
    session: AsyncSession,
    user: User,
    *,
    api_id: int,
    api_hash: str,
    session_string: str,
    phone: str,
    account_label: str | None,
) -> None:
    payload = TelegramSession(
        user_id=user.id,
        api_id=api_id,
        api_hash_enc=encrypt(api_hash),
        session_string_enc=encrypt(session_string),
        phone=phone,
        account_label=account_label,
    )
    await session.merge(payload)


async def load_telegram_session(
    session: AsyncSession, user: User
) -> tuple[int, str, str] | None:
    row = await session.get(TelegramSession, user.id)
    if row is None:
        return None
    return row.api_id, decrypt(row.api_hash_enc), decrypt(row.session_string_enc)


async def delete_telegram_session(session: AsyncSession, user: User) -> None:
    row = await session.get(TelegramSession, user.id)
    if row is not None:
        await session.delete(row)
        await session.flush()


async def get_self_profile(session: AsyncSession, user: User) -> SelfProfile | None:
    """Возвращает self-profile владельца или None."""
    result = await session.execute(
        select(SelfProfile).where(SelfProfile.user_id == user.id)
    )
    return result.scalar_one_or_none()


async def count_new_personal_facts_since(
    session: AsyncSession,
    owner: User,
    since: datetime | None,
) -> int:
    """Count personal Memory facts created after `since`.

    Personal facts are those with contact_id IS NULL and is_active=True.
    If since is None, counts ALL active personal facts.
    """
    from sqlalchemy import func

    from src.db.models import Memory

    q = (
        select(func.count())
        .select_from(Memory)
        .where(
            Memory.user_id == owner.id,
            Memory.contact_id.is_(None),
            Memory.is_active,
        )
    )
    if since is not None:
        q = q.where(Memory.created_at > since)

    result = await session.execute(q)
    return result.scalar_one()


async def upsert_self_profile(
    session: AsyncSession, user: User, **kwargs: object
) -> SelfProfile:
    """Создаёт или обновляет self-profile владельца.

    Списки/словари автоматически сериализуются в JSON (Text колонки).
    Переданные ``**kwargs`` применяются только если значение не None.
    """
    import json

    serialized: dict[str, object] = {}
    for k, v in kwargs.items():
        if v is not None:
            serialized[k] = (
                json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
            )
    profile = await get_self_profile(session, user)
    if profile is None:
        profile = SelfProfile(user_id=user.id, **serialized)
        session.add(profile)
    else:
        for k, v in serialized.items():
            setattr(profile, k, v)
    await session.flush()
    return profile


async def get_persona(session: AsyncSession, user: User) -> AdaptivePersona:
    """Возвращает AdaptivePersona для пользователя, создаёт с дефолтами если нет."""
    r = await session.execute(
        select(AdaptivePersona).where(AdaptivePersona.user_id == user.id)
    )
    persona = r.scalar_one_or_none()
    if persona is None:
        persona = AdaptivePersona(user_id=user.id)
        session.add(persona)
        await session.flush()
    return persona


async def update_persona(session: AsyncSession, persona: AdaptivePersona, **kwargs):
    """Обновляет поля AdaptivePersona и проставляет updated_at."""
    for k, v in kwargs.items():
        if hasattr(persona, k):
            setattr(persona, k, v)
    persona.updated_at = datetime.now(timezone.utc)
    await session.flush()
