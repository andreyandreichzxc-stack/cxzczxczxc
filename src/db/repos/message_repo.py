"""Message repository — Message, ConversationState, AutoReplyLog, TranscriptionCache."""

from __future__ import annotations

import logging

from sqlalchemy import func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import (
    AutoReplyLog,
    ConversationState,
    Message,
    TranscriptionCache,
)

logger = logging.getLogger(__name__)


async def list_active_conversations(
    session: AsyncSession, user, status: str = "active", limit: int = 50
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
    date,
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
    user,
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
    user,
    peer_id: int,
) -> int:
    """Возвращает общее количество сообщений в чате с peer_id для данного пользователя."""
    result = await session.execute(
        select(func.count())
        .select_from(Message)
        .where(Message.user_id == user.id, Message.peer_id == peer_id)
    )
    return result.scalar_one()


async def fetch_my_messages_in_chat(
    session: AsyncSession,
    user,
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
    await session.flush()


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
    await session.flush()


async def list_recent_auto_replies(
    session: AsyncSession,
    user,
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


async def fetch_my_messages_global(
    session: AsyncSession,
    user,
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


async def upsert_conversation_state(
    session: AsyncSession,
    user,
    peer_id: int,
    *,
    status: str | None = None,
    increment_unread: bool = False,
    last_incoming_at=None,
    last_outgoing_at=None,
    last_auto_reply_at=None,
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
    user,
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
