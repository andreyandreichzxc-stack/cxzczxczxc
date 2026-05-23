from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, distinct, func, or_, select, text as sql_text
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.crypto import decrypt, encrypt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.actions.vector_store import VectorStore
from src.db.models import (
    AdaptivePersona,
    ApiKey,
    AutoReplyLog,
    Commitment,
    Contact,
    ContactProfile,
    ConversationState,
    Folder,
    LlmKeySlot,
    Memory,
    MemoryCandidate,
    MemoryCluster,
    MemoryClusterMember,
    MemoryLink,
    Message,
    NewsTopic,
    PendingAction,
    SelfProfile,
    Skill,
    SkillUsage,
    TelegramSession,
    Trajectory,
    TranscriptionCache,
    User,
    UserSettings,
)

logger = logging.getLogger(__name__)


# Фоновые таски (digest, news, reminders, auto_sync) одновременно тыкаются в
# get_or_create_user на пустой БД → UNIQUE race. Сериализуем процесс-локом.
_user_lock = asyncio.Lock()


async def get_or_create_user(
    session: AsyncSession, telegram_id: int, *, use_cache: bool = True
) -> User:
    if use_cache:
        from src.core.context_cache import get as cache_get

        cached = await cache_get(f"user:{telegram_id}")
        if cached is not None:
            # cached is user.id (int) — retrieve fresh session-attached object
            user = await session.get(User, cached)
            if user is not None:
                return user
    async with _user_lock:
        result = await session.execute(
            select(User).where(User.telegram_id == telegram_id)
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


async def upsert_api_key(
    session: AsyncSession, user: User, provider: str, key: str
) -> None:
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


async def get_api_key(session: AsyncSession, user: User, provider: str) -> str | None:
    """Возвращает сохранённый ключ(и). Если ключей несколько — через запятую."""
    result = await session.execute(
        select(ApiKey).where(ApiKey.user_id == user.id, ApiKey.provider == provider)
    )
    row = result.scalar_one_or_none()
    return decrypt(row.key_enc) if row is not None else None


async def get_api_keys(session: AsyncSession, user: User, provider: str) -> list[str]:
    """Возвращает список ключей для провайдера."""
    raw = await get_api_key(session, user, provider)
    if not raw:
        return []
    return [k.strip() for k in raw.split(",") if k.strip()]


# ─── LlmKeySlot CRUD ────────────────────────────────────────────────


async def add_key_slot(
    session: AsyncSession,
    user: User,
    provider: str,
    key: str,
    *,
    purpose: str = "main",
    label: str | None = None,
    priority: int = 0,
) -> tuple[LlmKeySlot, bool]:
    """Добавляет слот ключа.

    Возвращает (LlmKeySlot, is_new):
      - is_new=True  — слот создан впервые
      - is_new=False — ключ с таким же значением уже существует (слот #N)
    """
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
        key_enc=encrypt(key),
        priority=priority,
    )
    session.add(slot)
    await session.flush()
    return slot, True


# ─── Learning loop: trajectories and skills ────────────────────────


async def add_trajectory(
    session: AsyncSession,
    user: User,
    *,
    request_text: str,
    route_mode: str | None = None,
    intent_json: dict | None = None,
    actions_json: list | None = None,
    used_skills_json: list | None = None,
    memory_ids_json: list | None = None,
    response_text: str | None = None,
    success: bool = True,
    error: str | None = None,
    latency_ms: int | None = None,
) -> Trajectory:
    row = Trajectory(
        user_id=user.id,
        request_text=request_text[:8000],
        route_mode=route_mode,
        intent_json=intent_json,
        actions_json=actions_json,
        used_skills_json=used_skills_json,
        memory_ids_json=memory_ids_json,
        response_text=response_text[:8000] if response_text else None,
        success=success,
        error=error[:4000] if error else None,
        latency_ms=latency_ms,
    )
    session.add(row)
    await session.flush()
    return row


async def list_trajectories(
    session: AsyncSession,
    user: User,
    *,
    only_errors: bool = False,
    limit: int = 20,
) -> list[Trajectory]:
    q = select(Trajectory).where(Trajectory.user_id == user.id)
    if only_errors:
        q = q.where(Trajectory.success.is_(False))
    q = q.order_by(Trajectory.created_at.desc()).limit(limit)
    r = await session.execute(q)
    return list(r.scalars().all())


async def upsert_skill(
    session: AsyncSession,
    user: User,
    *,
    name: str,
    description: str | None = None,
    trigger_patterns_json: list | None = None,
    body: str,
    enabled: bool = True,
    review_status: str = "approved",
) -> Skill:
    # Feature 3: YAML frontmatter parsing
    # Если description содержит YAML frontmatter (---...---),
    # парсим метаданные и сохраняем в trigger_patterns_json как __yaml__
    clean_description = description
    yaml_metadata: dict[str, object] = {}
    if description and description.strip().startswith("---"):
        try:
            from src.core.intelligence.skill_yaml import extract_frontmatter_metadata

            yaml_metadata, clean_description = extract_frontmatter_metadata(description)
        except Exception:
            logger.debug("upsert_skill: YAML frontmatter parse skipped", exc_info=True)
            clean_description = description

    # Собираем trigger_patterns_json: базовые паттерны + YAML метаданные
    patterns = list(trigger_patterns_json or [])
    if yaml_metadata:
        # Добавляем теги из YAML как паттерны
        tags = yaml_metadata.get("tags", [])
        if isinstance(tags, list):
            for tag in tags:
                if isinstance(tag, str) and tag.strip() not in patterns:
                    patterns.append(tag.strip())
        # Сохраняем структурированные метаданные как __yaml__
        # Убираем дубликат __yaml__ если уже есть
        patterns = [
            p for p in patterns if not (isinstance(p, dict) and "__yaml__" in p)
        ]
        patterns.append({"__yaml__": yaml_metadata})

    result = await session.execute(
        select(Skill).where(
            Skill.user_id == user.id,
            func.lower(Skill.name) == name.lower().strip(),
        )
    )
    skill = result.scalar_one_or_none()
    if skill is None:
        skill = Skill(
            user_id=user.id,
            name=name.strip(),
            description=clean_description,
            trigger_patterns_json=patterns,
            body=body,
            enabled=enabled,
            review_status=review_status,
        )
        session.add(skill)
    else:
        skill.description = clean_description
        skill.trigger_patterns_json = patterns
        skill.body = body
        skill.enabled = enabled
        skill.review_status = review_status
        skill.updated_at = datetime.now(timezone.utc)
    await session.flush()
    return skill


async def list_skills(
    session: AsyncSession,
    user: User,
    *,
    enabled: bool | None = None,
    review_status: str | None = None,
    limit: int = 50,
) -> list[Skill]:
    q = select(Skill).where(Skill.user_id == user.id)
    if enabled is not None:
        q = q.where(Skill.enabled == enabled)
    if review_status:
        q = q.where(Skill.review_status == review_status)
    q = q.order_by(Skill.success_count.desc(), Skill.updated_at.desc()).limit(limit)
    r = await session.execute(q)
    return list(r.scalars().all())


async def get_skill_by_name(
    session: AsyncSession, user: User, name: str
) -> Skill | None:
    r = await session.execute(
        select(Skill).where(
            Skill.user_id == user.id,
            func.lower(Skill.name) == name.lower().strip(),
        )
    )
    return r.scalar_one_or_none()


async def set_skill_enabled(
    session: AsyncSession,
    user: User,
    name: str,
    enabled: bool,
    *,
    review_status: str | None = None,
) -> Skill | None:
    skill = await get_skill_by_name(session, user, name)
    if skill is None:
        return None
    skill.enabled = enabled
    if review_status is not None:
        skill.review_status = review_status
    skill.updated_at = datetime.now(timezone.utc)
    await session.flush()
    return skill


async def add_skill_usage(
    session: AsyncSession,
    user: User,
    skill: Skill,
    *,
    trajectory_id: int | None = None,
    success: bool = True,
) -> SkillUsage:
    usage = SkillUsage(
        user_id=user.id,
        skill_id=skill.id,
        trajectory_id=trajectory_id,
        success=success,
    )
    session.add(usage)
    if success:
        skill.success_count = (skill.success_count or 0) + 1
    else:
        skill.failure_count = (skill.failure_count or 0) + 1
    skill.last_used_at = datetime.now(timezone.utc)
    skill.updated_at = datetime.now(timezone.utc)
    await session.flush()
    return usage


async def list_key_slots(
    session: AsyncSession,
    user: User,
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
    user: User,
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


# ─── Contacts ────────────────────────────────────────────────────────


async def upsert_contact(
    session: AsyncSession,
    user: User,
    *,
    peer_id: int,
    peer_kind: str,
    display_name: str,
    username: str | None = None,
    phone: str | None = None,
    is_bot: bool = False,
    is_archived: bool | None = None,
    folder_names: str | None = None,
) -> Contact:
    result = await session.execute(
        select(Contact).where(Contact.user_id == user.id, Contact.peer_id == peer_id)
    )
    contact = result.scalar_one_or_none()
    if contact is None:
        contact = Contact(
            user_id=user.id,
            peer_id=peer_id,
            peer_kind=peer_kind,
            is_bot=is_bot,
            is_archived=bool(is_archived) if is_archived is not None else False,
            display_name=display_name,
            username=username,
            phone=phone,
            folder_names=folder_names,
        )
        session.add(contact)
        await session.flush()
    else:
        contact.peer_kind = peer_kind
        contact.is_bot = is_bot
        if is_archived is not None:
            contact.is_archived = is_archived
        contact.display_name = display_name
        contact.username = username
        contact.phone = phone
        if folder_names is not None:
            contact.folder_names = folder_names
    return contact


async def list_contacts(
    session: AsyncSession,
    user: User,
    *,
    kinds: tuple[str, ...] | None = None,
    include_bots: bool = False,
    only_news_sources: bool = False,
    include_archived: bool | None = None,
) -> list[Contact]:
    # include_archived=None → берём решение из настроек пользователя
    if include_archived is None:
        include_archived = not user.settings.ignore_archived if user.settings else False

    query = select(Contact).where(Contact.user_id == user.id)
    if kinds:
        query = query.where(Contact.peer_kind.in_(kinds))
    if not include_bots:
        query = query.where(Contact.is_bot.is_(False))
    if only_news_sources:
        query = query.where(Contact.is_news_source.is_(True))
    if not include_archived:
        query = query.where(Contact.is_archived.is_(False))
    result = await session.execute(query)
    return list(result.scalars().all())


async def set_news_source(
    session: AsyncSession, user: User, peer_id: int, value: bool
) -> bool:
    result = await session.execute(
        select(Contact).where(Contact.user_id == user.id, Contact.peer_id == peer_id)
    )
    contact = result.scalar_one_or_none()
    if contact is None:
        return False
    contact.is_news_source = value
    return True


async def get_contact(
    session: AsyncSession, user: User, peer_id: int
) -> Contact | None:
    result = await session.execute(
        select(Contact).where(Contact.user_id == user.id, Contact.peer_id == peer_id)
    )
    return result.scalar_one_or_none()


async def list_active_conversations(
    session: AsyncSession, user: User, status: str = "active", limit: int = 50
) -> list[ConversationState]:
    result = await session.execute(
        select(ConversationState)
        .where(ConversationState.user_id == user.id, ConversationState.status == status)
        .order_by(ConversationState.last_incoming_at.desc().nullslast())
        .limit(limit)
    )
    return list(result.scalars().all())


async def upsert_message(
    session: AsyncSession,
    *,
    user_id: int,
    peer_id: int,
    message_id: int,
    sender_id: int | None,
    sender_name: str | None,
    is_outgoing: bool,
    date: datetime,
    kind: str,
    text: str | None,
    transcript: str | None = None,
    media_path: str | None = None,
    extracted_text: str | None = None,
) -> None:
    stmt = sqlite_insert(Message).values(
        user_id=user_id,
        peer_id=peer_id,
        message_id=message_id,
        sender_id=sender_id,
        sender_name=sender_name,
        is_outgoing=is_outgoing,
        date=date,
        kind=kind,
        text=text,
        transcript=transcript,
        media_path=media_path,
        extracted_text=extracted_text,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["user_id", "peer_id", "message_id"],
        set_={
            "text": stmt.excluded.text,
            "transcript": func.coalesce(stmt.excluded.transcript, Message.transcript),
            "extracted_text": func.coalesce(
                stmt.excluded.extracted_text, Message.extracted_text
            ),
            "media_path": func.coalesce(stmt.excluded.media_path, Message.media_path),
            "kind": stmt.excluded.kind,
            "sender_name": func.coalesce(
                stmt.excluded.sender_name, Message.sender_name
            ),
        },
    )
    await session.execute(stmt)


async def fetch_chat_messages(
    session: AsyncSession,
    user: User,
    peer_id: int,
    limit: int = 50,
) -> list[Message]:
    result = await session.execute(
        select(Message)
        .where(Message.user_id == user.id, Message.peer_id == peer_id)
        .order_by(Message.date.desc())
        .limit(limit)
    )
    return list(reversed(result.scalars().all()))


async def count_messages(
    session: AsyncSession,
    user: User,
    peer_id: int,
) -> int:
    """Возвращает общее количество сообщений в чате с peer_id для данного пользователя."""
    result = await session.execute(
        select(func.count())
        .select_from(Message)
        .where(Message.user_id == user.id, Message.peer_id == peer_id)
    )
    return result.scalar_one()


async def get_watched_peers(session: AsyncSession, user: User) -> set[int]:
    """Возвращает множество peer_id отслеживаемых чатов."""
    raw = user.settings.watched_peers
    if not raw:
        return set()
    try:
        parsed = json.loads(raw)
        return set(int(p) for p in parsed)
    except (json.JSONDecodeError, TypeError, ValueError):
        return set()


async def is_peer_watched(session: AsyncSession, user: User, peer_id: int) -> bool:
    """Проверяет, отслеживается ли чат peer_id."""
    watched = await get_watched_peers(session, user)
    return peer_id in watched


async def add_watched_peer(session: AsyncSession, user: User, peer_id: int) -> None:
    """Добавляет peer_id в список отслеживаемых."""
    async with _user_lock:
        watched = await get_watched_peers(session, user)
        watched.add(peer_id)
        user.settings.watched_peers = json.dumps(sorted(watched))
        await session.flush()


async def remove_watched_peer(session: AsyncSession, user: User, peer_id: int) -> None:
    """Удаляет peer_id из списка отслеживаемых."""
    async with _user_lock:
        watched = await get_watched_peers(session, user)
        watched.discard(peer_id)
        user.settings.watched_peers = json.dumps(sorted(watched)) if watched else None
        await session.flush()


@dataclass
class FtsHit:
    user_id: int
    peer_id: int
    message_id: int
    sender_name: str | None
    snippet: str
    rank: float
    peer_name: str | None = None
    date: datetime | None = None


def _fts_query_for(query: str) -> str:
    """Build an FTS5-safe MATCH expression from free-text user query.

    Each word becomes a prefix-match joined with OR.
    FTS5 operator keywords (OR, AND, NOT, NEAR) are double-quoted to
    prevent them from being interpreted as query operators — this is the
    standard SQLite FTS5 escaping mechanism for literal keyword search.
    """
    _FTS5_KEYWORDS = frozenset({"or", "and", "not", "near"})

    parts: list[str] = []
    for raw in query.split():
        clean = "".join(ch for ch in raw if ch.isalnum() or ch in "_-")
        if len(clean) < 2:
            continue
        lower = clean.lower()
        if lower in _FTS5_KEYWORDS:
            parts.append(f'"{lower}"')
        else:
            parts.append(lower + "*")
    if not parts:
        return ""
    return " OR ".join(parts)


async def fts_search(
    session: AsyncSession,
    user_id: int,
    query: str,
    *,
    limit: int = 50,
) -> list[FtsHit]:
    fts_q = _fts_query_for(query)
    if not fts_q:
        return []
    sql = """
        SELECT m.user_id, m.peer_id, m.message_id, m.sender_name,
               snippet(messages_fts, -1, '', '', '…', 16) AS snippet,
               bm25(messages_fts) AS rank,
               c.display_name AS peer_name,
               m.date
        FROM messages_fts
        JOIN messages m ON m.id = messages_fts.rowid
        LEFT JOIN contacts c ON c.user_id = m.user_id AND c.peer_id = m.peer_id
        WHERE messages_fts MATCH :q AND m.user_id = :uid
        ORDER BY rank
        LIMIT :lim
    """
    result = await session.execute(
        sql_text(sql),
        {"q": fts_q, "uid": user_id, "lim": limit},
    )
    rows = result.mappings().all()
    return [
        FtsHit(
            user_id=int(r["user_id"]),
            peer_id=int(r["peer_id"]),
            message_id=int(r["message_id"]),
            sender_name=r["sender_name"],
            snippet=r["snippet"] or "",
            rank=float(r["rank"]) if r["rank"] is not None else 0.0,
            peer_name=r["peer_name"],
            date=r["date"],
        )
        for r in rows
    ]


async def cross_chat_search(
    session: AsyncSession,
    user: User,
    query: str,
    limit: int = 30,
) -> dict[int, list[FtsHit]]:
    """
    Поиск по всем чатам, группировка по peer_id.
    Возвращает {peer_id: [FtsHit, ...]}.
    """
    hits = await fts_search(session, user.id, query, limit=limit)
    result: dict[int, list[FtsHit]] = {}
    for hit in hits:
        result.setdefault(hit.peer_id, []).append(hit)
    return result


async def fetch_my_messages_in_chat(
    session: AsyncSession,
    user: User,
    peer_id: int,
    limit: int = 100,
) -> list[Message]:
    result = await session.execute(
        select(Message)
        .where(
            Message.user_id == user.id,
            Message.peer_id == peer_id,
            Message.is_outgoing.is_(True),
        )
        .order_by(Message.date.desc())
        .limit(limit)
    )
    return list(reversed(result.scalars().all()))


async def get_cached_transcript(session: AsyncSession, file_id: str) -> str | None:
    row = await session.get(TranscriptionCache, file_id)
    return row.text if row else None


async def cache_transcript(
    session: AsyncSession,
    file_id: str,
    text: str,
    duration_seconds: float | None = None,
) -> None:
    existing = await session.get(TranscriptionCache, file_id)
    if existing is None:
        session.add(
            TranscriptionCache(
                file_id=file_id, text=text, duration_seconds=duration_seconds
            )
        )
    else:
        existing.text = text
        existing.duration_seconds = duration_seconds


async def add_commitment(
    session: AsyncSession,
    *,
    user_id: int,
    peer_id: int,
    peer_name: str | None,
    message_id: int | None,
    direction: str,
    text: str,
    deadline_at: datetime | None,
    source_memory_id: int | None = None,
) -> Commitment:
    c = Commitment(
        user_id=user_id,
        peer_id=peer_id,
        peer_name=peer_name,
        message_id=message_id,
        direction=direction,
        text=text,
        deadline_at=deadline_at,
        source_memory_id=source_memory_id,
    )
    session.add(c)
    await session.flush()
    return c


async def list_open_commitments(
    session: AsyncSession,
    user: User,
    *,
    direction: str | None = None,
) -> list[Commitment]:
    query = select(Commitment).where(
        Commitment.user_id == user.id,
        Commitment.status == "open",
    )
    if direction:
        query = query.where(Commitment.direction == direction)
    query = query.order_by(
        Commitment.deadline_at.is_(None), Commitment.deadline_at.asc()
    )
    result = await session.execute(query)
    return list(result.scalars().all())


async def update_commitment_status(
    session: AsyncSession, commitment_id: int, status: str
) -> None:
    c = await session.get(Commitment, commitment_id)
    if c is not None:
        c.status = status
        await session.flush()


async def get_commitment(
    session: AsyncSession, commitment_id: int
) -> Commitment | None:
    return await session.get(Commitment, commitment_id)


async def get_commitment_by_source_memory(
    session: AsyncSession, user_id: int, source_memory_id: int
) -> Commitment | None:
    result = await session.execute(
        select(Commitment).where(
            Commitment.user_id == user_id,
            Commitment.source_memory_id == source_memory_id,
        )
    )
    return result.scalar_one_or_none()


async def add_auto_reply_log(
    session: AsyncSession,
    *,
    user_id: int,
    peer_id: int,
    peer_name: str | None,
    incoming_text: str | None,
    reply_text: str,
) -> None:
    session.add(
        AutoReplyLog(
            user_id=user_id,
            peer_id=peer_id,
            peer_name=peer_name,
            incoming_text=incoming_text,
            reply_text=reply_text,
        )
    )


async def list_recent_auto_replies(
    session: AsyncSession,
    user: User,
    *,
    limit: int = 10,
) -> list[AutoReplyLog]:
    result = await session.execute(
        select(AutoReplyLog)
        .where(AutoReplyLog.user_id == user.id)
        .order_by(AutoReplyLog.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def create_pending_action(
    session: AsyncSession,
    *,
    user_id: int,
    kind: str,
    payload: str,
) -> PendingAction:
    pa = PendingAction(user_id=user_id, kind=kind, payload=payload)
    session.add(pa)
    await session.flush()
    return pa


async def get_pending_action(
    session: AsyncSession, action_id: int, user: User
) -> PendingAction | None:
    result = await session.execute(
        select(PendingAction).where(
            PendingAction.id == action_id, PendingAction.user_id == user.id
        )
    )
    return result.scalar_one_or_none()


async def delete_pending_action(
    session: AsyncSession, action_id: int, user: User
) -> None:
    pa = await session.get(PendingAction, action_id)
    if pa is not None and pa.user_id == user.id:
        await session.delete(pa)


async def list_news_topics(
    session: AsyncSession,
    user: User,
    *,
    only_enabled: bool = False,
) -> list[NewsTopic]:
    query = (
        select(NewsTopic)
        .where(NewsTopic.user_id == user.id)
        .order_by(NewsTopic.created_at.asc())
    )
    if only_enabled:
        query = query.where(NewsTopic.enabled.is_(True))
    result = await session.execute(query)
    return list(result.scalars().all())


async def add_news_topic(
    session: AsyncSession,
    user: User,
    topic: str,
    *,
    hours: int = 24,
) -> NewsTopic:
    nt = NewsTopic(user_id=user.id, topic=topic.strip(), hours=hours)
    session.add(nt)
    await session.flush()
    return nt


async def delete_news_topic(session: AsyncSession, user: User, topic_id: int) -> bool:
    nt = await session.get(NewsTopic, topic_id)
    if nt is None or nt.user_id != user.id:
        return False
    await session.delete(nt)
    return True


async def toggle_news_topic(
    session: AsyncSession, user: User, topic_id: int
) -> bool | None:
    nt = await session.get(NewsTopic, topic_id)
    if nt is None or nt.user_id != user.id:
        return None
    nt.enabled = not nt.enabled
    return nt.enabled


async def add_memory(
    session: AsyncSession,
    user: User,
    *,
    fact: str,
    contact_id: int | None = None,
    sentiment: str | None = None,
    source: str = "chat",
    confidence: float = 0.5,
    message_id: int | None = None,
    cluster_topic: str | None = None,
    deduplicate: bool = True,
    embedding: list[float] | None = None,
    vector_store_obj: "VectorStore | None" = None,
    importance: float | None = None,
    decay_rate: float | None = None,
    memory_tier: int = 1,
    memory_type: str | None = None,
    pinned: bool = False,
    expires_at: datetime | None = None,
    use_count: int = 0,
) -> Memory | None:
    """
    Добавляет факт в память с дедупликацией.

    Два уровня дедупликации (при deduplicate=True):
      1. SHA256 хеш — точные повторы.
      2. Если передан embedding + vector_store_obj — семантическая
         дедупликация через Qdrant с динамическим порогом:
           - 0.92 — тот же source, возраст <7 дней (строже)
           - 0.78 — разные source (мягче)
           - 0.85 — остальные случаи

    При обнаружении дубликата повышает confidence (вес от source)
    и times_mentioned. Если факт содержит временные маркеры
    ("сейчас", "раньше", "уже не", "больше не", "перестал") —
    всегда создаётся новая запись.
    Если embedding передан, индексирует факт в Qdrant для будущих проверок.
    """
    from src.core.actions.stats_cache import invalidate

    fact = fact.strip()
    if len(fact) < 3:
        return None

    # Хеш для дедупликации (первые 64 бита SHA256)
    emb_hash = hashlib.sha256(fact.lower().strip().encode()).hexdigest()[:16]

    # Вес source для повышения confidence при мерже
    source_weight = {"chat": 0.3, "user": 0.6, "weekly": 0.15}.get(source, 0.3)

    # Временные маркеры — не мерджим, создаём как новый факт
    temporal_markers = {"сейчас", "раньше", "уже не", "больше не", "перестал"}
    has_temporal_marker = any(m in fact.lower() for m in temporal_markers)

    if deduplicate and not has_temporal_marker:
        # --- Уровень 1: SHA256 хеш (точные повторы) ---
        result = await session.execute(
            select(Memory)
            .where(
                Memory.user_id == user.id,
                Memory.embedding_hash == emb_hash,
            )
            .limit(1)
        )
        existing = result.scalar_one_or_none()
        if existing:
            existing.times_mentioned = (existing.times_mentioned or 1) + 1
            existing.confidence = min(1.0, existing.confidence + source_weight)
            existing.updated_at = datetime.now(timezone.utc)
            if sentiment and existing.sentiment != sentiment:
                existing.sentiment = "contradictory"  # маркируем противоречие
            await session.flush()
            await invalidate("mem_")
            return existing

        # --- Уровень 2: семантическая дедупликация через Qdrant ---
        if embedding is not None and vector_store_obj is not None:
            # Проверяем кэш эмбеддингов (на случай если embed уже закэширован)

            # Ищем кандидатов с запасом (порог 0.7)
            similar = await vector_store_obj.search_similar_memories(
                user_id=user.id,
                embedding=embedding,
                threshold=0.7,
                limit=3,
            )
            if similar:
                best = similar[0]
                existing = await session.get(Memory, best["memory_id"])
                if existing and existing.user_id == user.id:
                    # Динамический порог
                    now = datetime.now(timezone.utc)
                    age_days = (
                        (now - existing.created_at).days if existing.created_at else 999
                    )
                    same_source = existing.source == source
                    if same_source and age_days < 7:
                        dyn_threshold = 0.92
                    elif not same_source:
                        dyn_threshold = 0.78
                    else:
                        dyn_threshold = 0.85

                    if best["score"] >= dyn_threshold:
                        existing.times_mentioned = (existing.times_mentioned or 1) + 1
                        existing.confidence = min(
                            1.0, existing.confidence + source_weight
                        )
                        existing.updated_at = now
                        if sentiment and existing.sentiment != sentiment:
                            existing.sentiment = "contradictory"
                        await session.flush()
                        await invalidate("mem_")
                        return existing

    mem = Memory(
        user_id=user.id,
        contact_id=contact_id,
        fact=fact,
        sentiment=sentiment,
        source=source,
        confidence=confidence,
        times_mentioned=1,
        message_id=message_id,
        is_active=True,
        cluster_topic=cluster_topic,
        embedding_hash=emb_hash,
        importance=importance if importance is not None else 0.5,
        decay_rate=decay_rate if decay_rate is not None else 0.07,
        memory_tier=memory_tier,
        memory_type=memory_type,
        pinned=pinned,
        expires_at=expires_at,
        use_count=use_count,
    )
    session.add(mem)
    await session.flush()

    # Индексируем эмбеддинг в Qdrant для будущей дедупликации
    if embedding is not None and vector_store_obj is not None:
        try:
            await vector_store_obj.upsert_memory(
                memory_id=mem.id,
                user_id=user.id,
                contact_id=contact_id,
                fact=fact,
                embedding=embedding,
            )
        except Exception:
            logger.exception("Failed to index memory embedding in Qdrant")

    await invalidate("mem_")
    return mem


async def list_memories(
    session: AsyncSession,
    user: User,
    *,
    contact_id: int | None = None,
) -> list[Memory]:
    query = (
        select(Memory)
        .where(Memory.user_id == user.id)
        .order_by(Memory.created_at.desc())
    )
    if contact_id is not None:
        query = query.where(Memory.contact_id == contact_id)
    result = await session.execute(query)
    return list(result.scalars().all())


async def delete_memory(session: AsyncSession, user: User, memory_id: int) -> bool:
    from src.core.actions.stats_cache import invalidate

    m = await session.get(Memory, memory_id)
    if m is None or m.user_id != user.id:
        return False
    await session.delete(m)
    await invalidate("mem_")
    return True


async def add_memory_candidate(
    session: AsyncSession,
    user: User,
    *,
    fact: str,
    contact_id: int | None = None,
    sentiment: str | None = None,
    memory_type: str | None = None,
    source: str = "chat",
    importance: float = 0.5,
    decay_rate: float = 0.07,
) -> MemoryCandidate:
    candidate = MemoryCandidate(
        user_id=user.id,
        contact_id=contact_id,
        fact=fact,
        sentiment=sentiment,
        memory_type=memory_type,
        source=source,
        importance=importance,
        decay_rate=decay_rate,
    )
    session.add(candidate)
    await session.flush()
    return candidate


async def list_memory_candidates(
    session: AsyncSession,
    user: User,
    limit: int = 20,
) -> list[MemoryCandidate]:
    result = await session.execute(
        select(MemoryCandidate)
        .where(MemoryCandidate.user_id == user.id)
        .order_by(MemoryCandidate.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def delete_memory_candidate(
    session: AsyncSession,
    user: User,
    candidate_id: int,
) -> bool:
    obj = await session.get(MemoryCandidate, candidate_id)
    if obj and obj.user_id == user.id:
        await session.delete(obj)
        return True
    return False


async def fetch_my_messages_global(
    session: AsyncSession,
    user: User,
    limit: int = 200,
) -> list[Message]:
    """Получить последние N исходящих сообщений владельца из всех чатов."""
    result = await session.execute(
        select(Message)
        .where(
            Message.user_id == user.id,
            Message.is_outgoing.is_(True),
            Message.text.isnot(None),
            Message.text != "",
        )
        .order_by(Message.date.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def get_agent_cache(session: AsyncSession, cache_key: str) -> str | None:
    """Получить кэш агента."""
    from src.db.models import AgentCache

    result = await session.execute(
        select(AgentCache).where(AgentCache.cache_key == cache_key)
    )
    row = result.scalar_one_or_none()
    if row:
        now = datetime.now(timezone.utc)
        # Handle both old naive and new aware datetimes
        try:
            age = (now - row.created_at).total_seconds()
        except TypeError:
            from datetime import timezone as tz

            age = (now - row.created_at.replace(tzinfo=tz.utc)).total_seconds()
        if age < row.ttl_seconds:
            return row.result_json
        await session.delete(row)
        await session.flush()
    return None


async def upsert_agent_cache(
    session: AsyncSession, cache_key: str, result_json: str, ttl_seconds: int
) -> None:
    """Сохранить/обновить кэш агента."""
    from src.db.models import AgentCache

    result = await session.execute(
        select(AgentCache).where(AgentCache.cache_key == cache_key)
    )
    row = result.scalar_one_or_none()
    if row:
        row.result_json = result_json
        row.created_at = datetime.now(timezone.utc)
        row.ttl_seconds = ttl_seconds
    else:
        session.add(
            AgentCache(
                cache_key=cache_key,
                result_json=result_json,
                ttl_seconds=ttl_seconds,
            )
        )
    await session.flush()


async def search_memories(
    session: AsyncSession,
    user: User,
    query: str,
    *,
    contact_id: int | None = None,
) -> list[Memory]:
    # Пробуем FTS5 сначала; если пусто — ILIKE fallback
    results = await search_memories_fts(session, user, query, contact_id=contact_id)
    if results:
        return results
    stmt = (
        select(Memory)
        .where(
            Memory.user_id == user.id,
            Memory.fact.icontains(query),
        )
        .order_by(Memory.created_at.desc())
    )
    if contact_id is not None:
        stmt = stmt.where(Memory.contact_id == contact_id)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def search_memories_fts(
    session: AsyncSession,
    user: User,
    query: str,
    *,
    contact_id: int | None = None,
    limit: int = 50,
) -> list[Memory]:
    """Полнотекстовый поиск по памяти через FTS5 с ранжированием по bm25().

    Использует _fts_query_for() для преобразования запроса в FTS5-safe формат.
    """
    fts_q = _fts_query_for(query)
    if not fts_q:
        return []

    base_sql = """
        SELECT m.id FROM memories_fts
        JOIN memories m ON m.id = memories_fts.rowid
        WHERE memories_fts MATCH :q AND m.user_id = :uid
    """
    if contact_id is not None:
        base_sql += " AND m.contact_id = :cid"
    base_sql += " ORDER BY bm25(memories_fts) LIMIT :lim"

    params = {"q": fts_q, "uid": user.id, "lim": limit}
    if contact_id is not None:
        params["cid"] = contact_id

    result = await session.execute(sql_text(base_sql), params)
    ids = [r[0] for r in result.fetchall()]
    if not ids:
        return []

    rows = await session.execute(select(Memory).where(Memory.id.in_(ids)))
    mem_map = {m.id: m for m in rows.scalars().all()}
    return [mem_map[mid] for mid in ids if mid in mem_map]


async def search_memories_fts_with_scores(
    session: AsyncSession,
    user: User,
    query: str,
    *,
    contact_id: int | None = None,
    limit: int = 20,
) -> list[tuple[int, float]]:
    """FTS5 keyword search on memories_fts returning (memory_id, bm25_score).

    Returns results sorted by BM25 rank (ascending — lower is better).
    This is the keyword counterpart to vector_store.search_similar_memories()
    for use in reciprocal rank fusion (RRF).
    """
    fts_query = _fts_query_for(query)
    if not fts_query:
        return []

    sql_parts = [
        "SELECT m.id, bm25(memories_fts) AS score",
        "FROM memories_fts",
        "JOIN memories m ON m.id = memories_fts.rowid",
        "WHERE memories_fts MATCH :q AND m.user_id = :uid",
    ]
    params: dict = {"q": fts_query, "uid": user.id}

    if contact_id is not None:
        sql_parts.append("AND m.contact_id = :cid")
        params["cid"] = contact_id

    sql_parts.append("ORDER BY score")
    sql_parts.append("LIMIT :lim")
    params["lim"] = limit

    sql = "\n".join(sql_parts)
    result = await session.execute(sql_text(sql), params)
    rows = result.all()

    # Return (memory_id, bm25_score) — lower BM25 = better match
    return [(int(r[0]), float(r[1])) for r in rows if r[1] is not None]


async def find_similar_memories(
    session: AsyncSession, user: User, fact: str, threshold: float = 0.7
) -> list[Memory]:
    """Поиск похожих фактов через ILIKE (падёт на векторный поиск когда Qdrant будет готов)."""
    # Упрощённо: поиск по подстроке с грубой оценкой сходства
    words = [w for w in fact.lower().split() if len(w) > 2]
    if not words:
        return []
    # Ищем факты где есть хотя бы 2 общих слова
    conditions = [Memory.fact.icontains(w) for w in words[:5]]
    result = await session.execute(
        select(Memory).where(Memory.user_id == user.id, or_(*conditions))
    )
    return list(result.scalars().all())


async def get_memory_stats(session: AsyncSession, user: User) -> dict:
    """Статистика по памяти (кэшируется на 5 минут)."""
    from src.core.actions.stats_cache import get_cached, set_cache

    cache_key = f"mem_stats:{user.id}"
    cached = await get_cached(cache_key)
    if cached is not None:
        return cached

    result = await session.execute(
        select(Memory).where(Memory.user_id == user.id, Memory.is_active)
    )
    memories = list(result.scalars().all())
    by_sentiment = {}
    for m in memories:
        s = m.sentiment or "neutral"
        by_sentiment[s] = by_sentiment.get(s, 0) + 1
    by_source = {}
    for m in memories:
        by_source[m.source] = by_source.get(m.source, 0) + 1
    by_tier = {"tier_1": 0, "tier_2": 0, "tier_3": 0}
    for m in memories:
        key = f"tier_{m.memory_tier}"
        by_tier[key] = by_tier.get(key, 0) + 1
    stats = {
        "total": len(memories),
        "by_sentiment": by_sentiment,
        "by_source": by_source,
        "by_tier": by_tier,
        "high_confidence": sum(1 for m in memories if m.confidence >= 0.8),
        "with_contact": sum(1 for m in memories if m.contact_id is not None),
    }
    await set_cache(cache_key, stats)
    return stats


async def upsert_memory_cluster(
    session: AsyncSession,
    user: User,
    topic: str,
    *,
    summary: str | None = None,
    fact_count: int | None = None,
) -> MemoryCluster:
    """Создаёт или возвращает существующий кластер по теме."""
    result = await session.execute(
        select(MemoryCluster).where(
            MemoryCluster.user_id == user.id,
            MemoryCluster.topic == topic.lower().strip(),
        )
    )
    cluster = result.scalar_one_or_none()
    if cluster is None:
        cluster = MemoryCluster(user_id=user.id, topic=topic.lower().strip())
        session.add(cluster)
    if summary is not None:
        cluster.summary = summary
    if fact_count is not None:
        cluster.fact_count = fact_count
    await session.flush()
    return cluster


async def list_memory_clusters(
    session: AsyncSession, user: User
) -> list[MemoryCluster]:
    """Список кластеров памяти."""
    result = await session.execute(
        select(MemoryCluster)
        .where(MemoryCluster.user_id == user.id)
        .order_by(MemoryCluster.fact_count.desc())
    )
    return list(result.scalars().all())


async def add_member(
    session: AsyncSession,
    user_id: int,
    memory_id: int,
    cluster_id: int,
    score: float = 0.5,
) -> None:
    """Добавляет факт в кластер."""
    m = MemoryClusterMember(
        user_id=user_id,
        memory_id=memory_id,
        cluster_id=cluster_id,
        relevance_score=score,
    )
    session.add(m)
    await session.flush()


async def get_cluster_members(
    session: AsyncSession,
    user: User,
    cluster_id: int,
    limit: int = 20,
) -> list[Memory]:
    """Факты кластера, отсортированы по relevance_score."""
    q = (
        select(Memory)
        .join(MemoryClusterMember, Memory.id == MemoryClusterMember.memory_id)
        .where(
            MemoryClusterMember.cluster_id == cluster_id,
            MemoryClusterMember.user_id == user.id,
            Memory.is_active,
        )
        .order_by(MemoryClusterMember.relevance_score.desc())
        .limit(limit)
    )
    r = await session.execute(q)
    return list(r.scalars().all())


async def list_clusters_for_contact(
    session: AsyncSession,
    user: User,
    contact_id: int | None = None,
) -> list:
    """Кластеры для контакта (или общие)."""
    q = (
        select(
            MemoryCluster,
            func.count(distinct(MemoryClusterMember.memory_id)).label("fact_count"),
        )
        .join(
            MemoryClusterMember,
            MemoryCluster.id == MemoryClusterMember.cluster_id,
        )
        .join(Memory, Memory.id == MemoryClusterMember.memory_id)
        .where(
            MemoryCluster.user_id == user.id,
            Memory.is_active,
        )
    )
    if contact_id is not None:
        q = q.where(Memory.contact_id == contact_id)
    q = (
        q.group_by(MemoryCluster.id)
        .order_by(func.count(distinct(MemoryClusterMember.memory_id)).desc())
        .limit(10)
    )
    r = await session.execute(q)
    return list(r.all())


async def upsert_folders(
    session: AsyncSession, user: User, folders_data: list[dict]
) -> int:
    """Сохраняет/обновляет папки. folders_data: [{'telegram_folder_id': int, 'title': str, 'emoji': str|None}]."""
    async with _user_lock:
        # Удалить старые папки этого пользователя
        await session.execute(delete(Folder).where(Folder.user_id == user.id))
        # Вставить новые
        saved = 0
        for f in folders_data:
            session.add(
                Folder(
                    user_id=user.id,
                    telegram_folder_id=f["telegram_folder_id"],
                    title=f["title"],
                    emoji=f.get("emoji"),
                )
            )
            saved += 1
        await session.flush()
    return saved


async def list_folders(session: AsyncSession, user: User) -> list[Folder]:
    """Возвращает список папок пользователя."""
    result = await session.execute(
        select(Folder).where(Folder.user_id == user.id).order_by(Folder.title)
    )
    return list(result.scalars().all())


async def upsert_conversation_state(
    session: AsyncSession,
    user: User,
    peer_id: int,
    *,
    status: str | None = None,
    increment_unread: bool = False,
    last_incoming_at: datetime | None = None,
    last_outgoing_at: datetime | None = None,
    last_auto_reply_at: datetime | None = None,
) -> ConversationState:
    """Создаёт или обновляет состояние диалога с контактом."""
    result = await session.execute(
        select(ConversationState).where(
            ConversationState.user_id == user.id,
            ConversationState.peer_id == peer_id,
        )
    )
    state = result.scalar_one_or_none()
    if state is None:
        state = ConversationState(
            user_id=user.id,
            peer_id=peer_id,
            status=status or "active",
            unread_count=1 if increment_unread else 0,
            last_incoming_at=last_incoming_at,
            last_outgoing_at=last_outgoing_at,
            last_auto_reply_at=last_auto_reply_at,
        )
        session.add(state)
    else:
        if status is not None:
            state.status = status
        if increment_unread:
            state.unread_count = (state.unread_count or 0) + 1
        if last_incoming_at is not None:
            state.last_incoming_at = last_incoming_at
        if last_outgoing_at is not None:
            state.last_outgoing_at = last_outgoing_at
        if last_auto_reply_at is not None:
            state.last_auto_reply_at = last_auto_reply_at
    await session.flush()
    return state


async def get_conversation_state(
    session: AsyncSession,
    user: User,
    peer_id: int,
) -> ConversationState | None:
    """Возвращает состояние диалога с контактом."""
    result = await session.execute(
        select(ConversationState).where(
            ConversationState.user_id == user.id,
            ConversationState.peer_id == peer_id,
        )
    )
    return result.scalar_one_or_none()


async def link_memories(
    session: AsyncSession,
    user: User,
    source_id: int,
    target_id: int,
    weight: float = 0.5,
    relation_type: str | None = None,
) -> MemoryLink | None:
    """Создать/обновить связь между фактами памяти (many-to-many)."""

    # Проверить что оба факта принадлежат пользователю
    result = await session.execute(
        select(Memory).where(
            Memory.id.in_([source_id, target_id]), Memory.user_id == user.id
        )
    )
    if len(result.scalars().all()) < 2:
        return None  # один из фактов не найден или чужой

    # Проверить существующую связь
    existing = await session.execute(
        select(MemoryLink).where(
            MemoryLink.user_id == user.id,
            MemoryLink.source_id == source_id,
            MemoryLink.target_id == target_id,
        )
    )
    existing = existing.scalar_one_or_none()
    if existing:
        existing.weight = weight
        if relation_type:
            existing.relation_type = relation_type
        await session.flush()
        return existing

    # Создать новую + обратную
    link = MemoryLink(
        user_id=user.id,
        source_id=source_id,
        target_id=target_id,
        weight=weight,
        relation_type=relation_type,
    )
    session.add(link)

    # Обратная связь (если не дубль)
    reverse_check = await session.execute(
        select(MemoryLink).where(
            MemoryLink.user_id == user.id,
            MemoryLink.source_id == target_id,
            MemoryLink.target_id == source_id,
        )
    )
    if not reverse_check.scalar_one_or_none():
        rev = MemoryLink(
            user_id=user.id,
            source_id=target_id,
            target_id=source_id,
            weight=weight,
            relation_type=relation_type,
        )
        session.add(rev)

    await session.flush()
    return link


async def unlink_memories(
    session: AsyncSession, user: User, source_id: int, target_id: int
) -> None:
    """Удалить связь между фактами (в обе стороны)."""
    from sqlalchemy import and_, or_

    from sqlalchemy import delete

    await session.execute(
        delete(MemoryLink).where(
            MemoryLink.user_id == user.id,
            or_(
                and_(
                    MemoryLink.source_id == source_id,
                    MemoryLink.target_id == target_id,
                ),
                and_(
                    MemoryLink.source_id == target_id,
                    MemoryLink.target_id == source_id,
                ),
            ),
        )
    )
    await session.flush()


async def get_linked_memories(
    session: AsyncSession, user: User, memory_id: int, limit: int = 10
) -> list[dict]:
    """Получить связанные факты с весами."""
    result = await session.execute(
        select(Memory, MemoryLink.weight, MemoryLink.relation_type)
        .join(MemoryLink, MemoryLink.target_id == Memory.id)
        .where(
            MemoryLink.source_id == memory_id,
            MemoryLink.user_id == user.id,
            Memory.is_active,
        )
        .order_by(MemoryLink.weight.desc())
        .limit(limit)
    )
    rows = result.all()
    linked: list[dict] = []
    for mem, weight, rel_type in rows:
        linked.append({"memory": mem, "weight": weight, "relation_type": rel_type})
    return linked


async def get_memory_graph(
    session: AsyncSession,
    user: User,
    memory_id: int,
    max_depth: int = 3,
    max_nodes: int = 20,
) -> list[dict]:
    """Строит граф связанных фактов BFS от memory_id.

    Оптимизация: вместо N запросов (на каждый узел) делаем 2 запроса:
    1) все MemoryLink пользователя → строим adjacency dict
    2) все Memory для посещённых ID → batch load
    """
    # ── Phase 1: Load ALL MemoryLinks for this user in ONE query ──────
    rows = (
        await session.execute(
            select(
                MemoryLink.source_id,
                MemoryLink.target_id,
                MemoryLink.weight,
                MemoryLink.relation_type,
            )
            .where(MemoryLink.user_id == user.id)
            .order_by(MemoryLink.weight.desc())
        )
    ).all()

    # Build in-memory adjacency dict: source_id -> [(target_id, weight, rel_type), ...]
    # Already sorted by weight DESC from the DB query
    adj: dict[int, list[tuple[int, float, str | None]]] = {}
    for source_id, target_id, weight, relation_type in rows:
        adj.setdefault(source_id, []).append((target_id, weight, relation_type))

    # ── Phase 2: BFS walk using the in-memory adjacency dict ─────────
    visited: set[int] = set()
    graph: list[dict] = []
    queue: list[tuple[int, int]] = [(memory_id, 0)]
    while queue and len(visited) < max_nodes:
        mid, depth = queue.pop(0)
        if mid in visited or depth > max_depth:
            continue
        visited.add(mid)
        if depth > 0:  # не добавляем корневой узел в граф, только соседей
            # Memory будет загружен в Phase 3 (batch)
            graph.append({"memory_id": mid, "depth": depth})
        if depth < max_depth:
            # adj.get(mid, []) уже отсортирован по weight DESC из Phase 1
            for target_id, weight, rel_type in adj.get(mid, [])[:10]:
                if target_id not in visited:
                    queue.append((target_id, depth + 1))

    if not graph:
        return []

    # ── Phase 3: Load ALL needed Memory objects in ONE batch query ───
    mem_ids = {entry["memory_id"] for entry in graph}
    result = await session.execute(select(Memory).where(Memory.id.in_(mem_ids)))
    mem_lookup: dict[int, Memory] = {m.id: m for m in result.scalars().all()}

    # ── Phase 4: Assemble the graph from the lookup dict ──────────────
    for entry in graph:
        mid = entry.pop("memory_id")
        mem = mem_lookup.get(mid)
        if mem:
            entry["memory"] = mem
        # если memory удалена между Phase 2 и Phase 3 — пропускаем
        # (аналогично оригинальному поведению `if mem:`)

    return graph


# ─── SelfProfile CRUD ────────────────────────────────────────────────


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


# ─── ContactProfile CRUD ─────────────────────────────────────────────


async def upsert_contact_profile(
    session: AsyncSession,
    user: User,
    contact_id: int,
    **kwargs: object,
) -> ContactProfile:
    """Создаёт или обновляет профиль контакта.

    Переданные ``**kwargs`` применяются только если значение не None.
    """
    result = await session.execute(
        select(ContactProfile).where(
            ContactProfile.user_id == user.id,
            ContactProfile.contact_id == contact_id,
        )
    )
    profile = result.scalar_one_or_none()
    if profile is None:
        profile = ContactProfile(user_id=user.id, contact_id=contact_id, **kwargs)
        session.add(profile)
    else:
        for k, v in kwargs.items():
            if v is not None:
                setattr(profile, k, v)
    await session.flush()
    return profile


async def get_contact_profile(
    session: AsyncSession,
    user: User,
    contact_id: int,
) -> ContactProfile | None:
    """Возвращает профиль контакта или None."""
    result = await session.execute(
        select(ContactProfile).where(
            ContactProfile.user_id == user.id,
            ContactProfile.contact_id == contact_id,
        )
    )
    return result.scalar_one_or_none()


async def list_contact_profiles(
    session: AsyncSession,
    user: User,
    limit: int = 50,
) -> list[ContactProfile]:
    """Возвращает профили контактов, отсортированные по близости (убывание)."""
    result = await session.execute(
        select(ContactProfile)
        .where(ContactProfile.user_id == user.id)
        .order_by(ContactProfile.closeness.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


# ─── AdaptivePersona CRUD ──────────────────────────────────────────


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
