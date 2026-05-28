"""Auto-logs agent-owner conversations to AgentSession table."""

from __future__ import annotations
import logging
from datetime import datetime, timezone
from src.db.session import get_session
from src.db.repo import get_or_create_user
from sqlalchemy import select
from src.db.models._session import AgentSession, AgentSessionMessage

logger = logging.getLogger(__name__)
_active_sessions: dict[int, int] = {}  # telegram_id → session_id


async def log_user_message(telegram_id: int, text: str) -> None:
    """Log user message to current session."""
    try:
        async with get_session() as session:
            owner = await get_or_create_user(session, telegram_id)
            sid = _active_sessions.get(telegram_id)
            if sid is None:
                new_sess = AgentSession(user_id=owner.id, session_type="chat")
                session.add(new_sess)
                await session.flush()
                sid = new_sess.id
                _active_sessions[telegram_id] = sid
            msg = AgentSessionMessage(session_id=sid, role="user", content=text[:2000])
            session.add(msg)
            # Update turn count
            sess = await session.get(AgentSession, sid)
            if sess:
                sess.turn_count = (sess.turn_count or 0) + 1
            await session.flush()
    except Exception:
        logger.debug("Session log failed (user message)", exc_info=True)


async def log_assistant_response(telegram_id: int, text: str) -> None:
    """Log assistant response to current session."""
    try:
        async with get_session() as session:
            owner = await get_or_create_user(session, telegram_id)
            sid = _active_sessions.get(telegram_id)
            if sid is None:
                new_sess = AgentSession(user_id=owner.id, session_type="chat")
                session.add(new_sess)
                await session.flush()
                sid = new_sess.id
                _active_sessions[telegram_id] = sid
            msg = AgentSessionMessage(
                session_id=sid, role="assistant", content=text[:2000]
            )
            session.add(msg)
            sess = await session.get(AgentSession, sid)
            if sess:
                sess.turn_count = (sess.turn_count or 0) + 1
                if sess.turn_count and sess.turn_count % 20 == 0:
                    # Auto-summarize every 20 turns
                    recent = await session.execute(
                        select(AgentSessionMessage.content)
                        .where(AgentSessionMessage.session_id == sid)
                        .order_by(AgentSessionMessage.created_at.desc())
                        .limit(20)
                    )
                    texts = [r[0] for r in recent.fetchall() if r[0]]
                    if texts:
                        sess.summary = "\n".join(texts)[:500]
                        sess.ended_at = datetime.now(timezone.utc)
            await session.flush()
    except Exception:
        logger.debug("Session log failed (assistant)", exc_info=True)


async def end_session(telegram_id: int) -> None:
    """Mark current session as ended."""
    sid = _active_sessions.pop(telegram_id, None)
    if sid:
        try:
            async with get_session() as session:
                sess = await session.get(AgentSession, sid)
                if sess:
                    sess.ended_at = datetime.now(timezone.utc)
                    await session.flush()
        except Exception:
            logger.debug("Session end failed", exc_info=True)
