"""Records conversation turns into AgentSession / AgentSessionMessage for replay.

Usage:
    from src.core.memory.session_recorder import record_turn, close_session, get_session_history

    # Record a user or assistant turn (non-blocking, safe to wrap in try/except)
    await record_turn(session, telegram_id, "user", "Hello!")
    await record_turn(session, telegram_id, "assistant", "Hi there!")

    # Close the active session for a user
    await close_session(telegram_id, session)

    # Retrieve session history
    history = await get_session_history(session, telegram_id, limit=5)
"""

import logging
import asyncio
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models._session import AgentSession, AgentSessionMessage
from src.db.repo import get_or_create_user

logger = logging.getLogger(__name__)

# In-memory cache of active sessions: {telegram_id: (session_id, started_at)}
_active_sessions: dict[int, tuple[int, datetime]] = {}
_active_sessions_lock: asyncio.Lock = asyncio.Lock()

SESSION_INACTIVITY_TIMEOUT = timedelta(minutes=30)


async def _resolve_user_id(session: AsyncSession, telegram_id: int) -> int:
    """Convert telegram_id to internal users.id via get_or_create_user.

    Cached internally by the repo layer, so repeated calls are fast.
    """
    owner = await get_or_create_user(session, telegram_id)
    return owner.id


async def record_turn(
    db_session: AsyncSession,
    telegram_id: int,
    role: str,  # "user" or "assistant"
    content: str,
    session_type: str = "chat",
) -> None:
    """Record a conversation turn. Creates a new session if none is active.

    Args:
        db_session: Active SQLAlchemy async session (will NOT be committed here —
                    caller's context manager handles commit).
        telegram_id: Telegram user ID (converted to internal user_id internally).
        role: "user" or "assistant".
        content: Message text (truncated to 4000 chars).
        session_type: Session type label (default "chat").
    """
    async with _active_sessions_lock:
        cached = _active_sessions.get(telegram_id)
        session_id = cached[0] if cached else None

        if session_id is not None:
            # Check inactivity from cached timestamp (no DB query needed)
            cached_started = cached[1] if cached else None
            if cached_started:
                elapsed = datetime.now(timezone.utc) - cached_started
                if elapsed > SESSION_INACTIVITY_TIMEOUT:
                    # Auto-close stale session via DB
                    result = await db_session.execute(
                        select(AgentSession).where(AgentSession.id == session_id)
                    )
                    agent_session = result.scalar_one_or_none()
                    if agent_session is not None:
                        agent_session.ended_at = datetime.now(timezone.utc)
                    _active_sessions.pop(telegram_id, None)
                    session_id = None

        if session_id is None:
            # Create a new session (still under lock to avoid race)
            user_id = await _resolve_user_id(db_session, telegram_id)
            agent_session = AgentSession(
                user_id=user_id,
                session_type=session_type,
                started_at=datetime.now(timezone.utc),
                turn_count=0,
            )
            db_session.add(agent_session)
            await db_session.flush()
            session_id = agent_session.id
            _active_sessions[telegram_id] = (session_id, agent_session.started_at)

    # Record the message
    msg = AgentSessionMessage(
        session_id=session_id,
        role=role,
        content=content[:4000],
        created_at=datetime.now(timezone.utc),
    )
    db_session.add(msg)

    # Increment turn count
    await db_session.execute(
        sa_update(AgentSession)
        .where(AgentSession.id == session_id)
        .values(turn_count=AgentSession.turn_count + 1)
    )


async def close_session(telegram_id: int, db_session: AsyncSession) -> None:
    """Close the active session for a user (set ended_at)."""
    async with _active_sessions_lock:
        cached = _active_sessions.pop(telegram_id, None)
    session_id = cached[0] if cached else None
    if session_id is not None:
        await db_session.execute(
            sa_update(AgentSession)
            .where(AgentSession.id == session_id)
            .values(ended_at=datetime.now(timezone.utc))
        )


async def close_stale_sessions(
    db_session: AsyncSession, max_age_hours: int = 24
) -> int:
    """Close sessions that have been open for too long (no ended_at set).

    Runs during the nightly dream cycle to clean up sessions that were
    never explicitly closed (e.g. due to crashes or incomplete cleanup).

    Args:
        db_session: Active SQLAlchemy async session.
        max_age_hours: Sessions older than this (by started_at) will be closed.

    Returns:
        Number of sessions closed.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    result = await db_session.execute(
        sa_update(AgentSession)
        .where(AgentSession.ended_at.is_(None), AgentSession.started_at < cutoff)
        .values(ended_at=datetime.now(timezone.utc))
    )
    return result.rowcount


async def get_session_history(
    db_session: AsyncSession,
    telegram_id: int,
    limit: int = 10,
    session_type: str = "chat",
) -> list[dict]:
    """Get recent sessions with their messages for a given user.

    Args:
        db_session: Active SQLAlchemy async session.
        telegram_id: Telegram user ID.
        limit: Max number of sessions to return.
        session_type: Filter by session type (default "chat").

    Returns:
        List of dicts with keys: session_id, started_at, ended_at,
        turn_count, summary, messages (list of {role, content, time}).
    """
    # Resolve telegram_id → internal user_id
    user_id = await _resolve_user_id(db_session, telegram_id)

    # Fetch recent sessions
    result = await db_session.execute(
        select(AgentSession)
        .where(
            AgentSession.user_id == user_id,
            AgentSession.session_type == session_type,
        )
        .order_by(AgentSession.started_at.desc())
        .limit(limit)
    )
    sessions = result.scalars().all()

    # Fetch messages per session (single batch query, not N+1)
    session_ids = [s.id for s in sessions]
    if session_ids:
        result = await db_session.execute(
            select(AgentSessionMessage)
            .where(AgentSessionMessage.session_id.in_(session_ids))
            .order_by(AgentSessionMessage.created_at)
        )
        all_messages = result.scalars().all()
    else:
        all_messages = []

    # Group by session_id
    messages_by_session: dict[int, list] = {}
    for msg in all_messages:
        messages_by_session.setdefault(msg.session_id, []).append(msg)

    formatted: list[dict] = []
    for s in sessions:
        msgs = messages_by_session.get(s.id, [])
        messages = [
            {
                "role": m.role,
                "content": m.content,
                "time": m.created_at.isoformat(),
            }
            for m in msgs
        ]
        formatted.append(
            {
                "session_id": s.id,
                "started_at": s.started_at.isoformat() if s.started_at else None,
                "ended_at": s.ended_at.isoformat() if s.ended_at else None,
                "turn_count": s.turn_count,
                "summary": s.summary,
                "messages": messages,
            }
        )

    return formatted
