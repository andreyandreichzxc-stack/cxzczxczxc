"""Key repository — ApiKey, LlmKeySlot."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import (
    ApiKey,
    LlmKeySlot,
)
from src.crypto import decrypt, encrypt

logger = logging.getLogger(__name__)


async def upsert_api_key(session: AsyncSession, user, provider: str, key: str) -> None:
    # Нормализация: поддерживается несколько ключей через запятую
    parts = [k.strip() for k in key.split(",") if k.strip()]
    if not parts:
        return
    normalized = ",".join(parts)
    result = await session.execute(
        select(ApiKey).where(ApiKey.user_id == user.id, ApiKey.provider == provider)
    )
    existing = result.scalar_one_or_none()
    enc = encrypt(normalized)
    if existing is None:
        session.add(ApiKey(user_id=user.id, provider=provider, key_enc=enc))
    else:
        existing.key_enc = enc

    # Унификация: также сохраняем в LlmKeySlot (новое хранилище)
    # Каждый ключ из списка — отдельный слот
    existing_slots = await list_key_slots(session, user, provider=provider)
    existing_keys: set[str] = set()
    for s in existing_slots:
        try:
            existing_keys.add(decrypt(s.key_enc))
        except Exception:
            continue

    for i, single_key in enumerate(parts):
        if single_key not in existing_keys:
            slot = LlmKeySlot(
                user_id=user.id,
                provider=provider,
                purpose="main",
                label=f"{provider}/main",
                key_enc=encrypt(single_key),
                priority=i,
            )
            session.add(slot)
        else:
            # Ключ уже есть в LlmKeySlot — не дублируем
            pass

    await session.flush()


async def get_api_key(session: AsyncSession, user, provider: str) -> str | None:
    """Возвращает сохранённый ключ(и). Если ключей несколько — через запятую."""
    result = await session.execute(
        select(ApiKey).where(ApiKey.user_id == user.id, ApiKey.provider == provider)
    )
    row = result.scalar_one_or_none()
    return decrypt(row.key_enc) if row is not None else None


async def get_api_keys(session: AsyncSession, user, provider: str) -> list[str]:
    """Возвращает список ключей для провайдера."""
    raw = await get_api_key(session, user, provider)
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]


async def add_key_slot(
    session: AsyncSession,
    user,
    provider: str,
    key: str,
    *,
    purpose: str = "main",
    label: str | None = None,
    priority: int = 0,
    endpoint: str | None = None,
    model: str | None = None,
    category: str = "llm",
) -> tuple[LlmKeySlot, bool]:
    """Добавляет слот ключа.

    Возвращает (LlmKeySlot, is_new):
      - is_new=True  — слот создан впервые
      - is_new=False — ключ с таким же значением уже существует (слот #N)
    """
    from src.db.repos.session_repo import _get_user_lock

    lock = _get_user_lock(user.id)
    async with lock:
        # Проверка дубликатов: расшифровываем все существующие слоты пользователя
        # и сравниваем с новым ключом
        existing_slots = await list_key_slots(
            session, user, provider=provider, purpose=purpose
        )
        for existing in existing_slots:
            try:
                existing_key = decrypt(existing.key_enc)
                if existing_key == key:
                    return existing, False
            except Exception:
                continue

        slot = LlmKeySlot(
            user_id=user.id,
            provider=provider,
            purpose=purpose,
            label=label,
            endpoint=endpoint,
            model=model,
            category=category,
            key_enc=encrypt(key),
            priority=priority,
        )
        session.add(slot)
        await session.flush()
        return slot, True


async def list_key_slots(
    session: AsyncSession,
    user,
    provider: str | None = None,
    purpose: str | None = None,
) -> list[LlmKeySlot]:
    """Список слотов с фильтрацией."""
    q = select(LlmKeySlot).where(LlmKeySlot.user_id == user.id)
    if provider:
        q = q.where(LlmKeySlot.provider == provider)
    if purpose:
        q = q.where(LlmKeySlot.purpose == purpose)
    q = q.order_by(LlmKeySlot.priority.desc())
    r = await session.execute(q)
    return list(r.scalars().all())


async def get_active_keys(
    session: AsyncSession,
    user,
    provider: str,
    purpose: str = "main",
) -> list[LlmKeySlot]:
    """Активные (enabled, не в кулдауне) ключи для провайдера и назначения."""
    now = datetime.now(timezone.utc)
    q = (
        select(LlmKeySlot)
        .where(
            LlmKeySlot.user_id == user.id,
            LlmKeySlot.provider == provider,
            LlmKeySlot.purpose == purpose,
            LlmKeySlot.enabled,
            or_(LlmKeySlot.cooldown_until.is_(None), LlmKeySlot.cooldown_until <= now),
        )
        .order_by(LlmKeySlot.priority.desc())
    )
    r = await session.execute(q)
    return list(r.scalars().all())


async def mark_key_failure(
    session: AsyncSession,
    slot_id: int,
    error_msg: str,
    cooldown_sec: int = 120,
) -> None:
    """Помечает ключ как упавший с кулдауном."""
    slot = await session.get(LlmKeySlot, slot_id)
    if slot:
        slot.failure_count = (slot.failure_count or 0) + 1
        slot.last_error = error_msg[:256]
        slot.last_error_at = datetime.now(timezone.utc)
        slot.cooldown_until = datetime.now(timezone.utc) + timedelta(
            seconds=cooldown_sec
        )
        await session.flush()


async def mark_key_used(session: AsyncSession, slot_id: int) -> None:
    """Инкремент счётчика использования."""
    slot = await session.get(LlmKeySlot, slot_id)
    if slot:
        slot.usage_count = (slot.usage_count or 0) + 1
        slot.cooldown_until = None
        slot.last_error = None
        await session.flush()
