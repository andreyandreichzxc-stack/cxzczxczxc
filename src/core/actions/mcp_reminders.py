"""mcp_reminders tool — registered via @tool decorator.

Provides reminder (commitment) management operations:

- **list** — list active (open) reminders for the user.
- **create** — create a new reminder with optional deadline.

Both actions rely on the ``commitments`` table and repo-layer functions
from ``src.db.repo``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.actions.tool_registry import tool
from src.db.repo import (
    add_commitment,
    get_or_create_user,
    list_open_commitments,
)
from src.db.session import get_session

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════
# Tool: mcp_reminders
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="mcp_reminders",
    description=(
        "Manage reminders (commitments). Supports two actions:\n"
        "- 'list' — return all active (open) reminders for the user.\n"
        "- 'create' — create a new reminder with optional ISO-deadline."
    ),
    category="reminder",
    risk="low",
    params={
        "action": "str — 'list' or 'create'",
        "text": "str — reminder text (required for action='create')",
        "deadline": "str|None — ISO 8601 datetime string (optional, for create)",
    },
)
async def mcp_reminders(
    action: str,
    text: str = "",
    deadline: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """Reminder (commitment) management tool.

    Args:
        action: ``"list"`` or ``"create"``.
        text: Reminder text (required when ``action="create"``).
        deadline: ISO 8601 datetime string (optional, for ``action="create"``).

    Keyword Args:
        user: Owner's Telegram ID (int, defaults to 0).
        session: Optional ``AsyncSession`` (a new one is created if omitted).

    Returns:
        A dict with the result data or an ``"error"`` key on failure.
    """
    _user_val = kwargs.get("user", 0)
    # user may be an int (telegram_id) or a User ORM object — normalise
    if hasattr(_user_val, "telegram_id"):
        telegram_id: int = _user_val.telegram_id
    else:
        telegram_id = int(_user_val)

    try:
        if action == "list":
            return await _list_reminders(telegram_id, kwargs.get("session"))
        elif action == "create":
            return await _create_reminder(
                telegram_id, text, deadline, kwargs.get("session")
            )
        else:
            return {"error": f"Unknown action {action!r}. Valid actions: list, create"}
    except Exception as exc:
        logger.exception("mcp_reminders(%r) failed", action)
        return {"error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════
# Action implementations
# ══════════════════════════════════════════════════════════════════════════


async def _list_reminders(
    telegram_id: int,
    session: AsyncSession | None,
) -> dict[str, Any]:
    """List open commitments for the user."""
    if session is None:
        async with get_session() as session:
            return await _do_list(session, telegram_id)
    return await _do_list(session, telegram_id)


async def _do_list(session: AsyncSession, telegram_id: int) -> dict[str, Any]:
    user = await get_or_create_user(session, telegram_id)
    commitments = await list_open_commitments(session, user)

    reminders = [
        {
            "id": c.id,
            "text": c.text,
            "deadline": (c.deadline_at.isoformat() if c.deadline_at else None),
            "status": c.status,
        }
        for c in commitments
    ]

    return {
        "ok": True,
        "reminders": reminders,
        "count": len(reminders),
    }


async def _create_reminder(
    telegram_id: int,
    text: str,
    deadline: str,
    session: AsyncSession | None,
) -> dict[str, Any]:
    """Create a new reminder (commitment)."""
    if not text or not text.strip():
        return {"error": "text parameter is required for action='create'"}

    # Parse optional ISO deadline
    deadline_at: datetime | None = None
    if deadline and deadline.strip():
        try:
            deadline_at = datetime.fromisoformat(deadline.strip())
            # Ensure timezone-aware — default to UTC if naive
            if deadline_at.tzinfo is None:
                deadline_at = deadline_at.replace(tzinfo=timezone.utc)
        except ValueError as exc:
            return {"error": f"Invalid deadline format (expected ISO 8601): {exc}"}

    if session is None:
        async with get_session() as session:
            return await _do_create(session, telegram_id, text.strip(), deadline_at)
    return await _do_create(session, telegram_id, text.strip(), deadline_at)


async def _do_create(
    session: AsyncSession,
    telegram_id: int,
    text: str,
    deadline_at: datetime | None,
) -> dict[str, Any]:
    user = await get_or_create_user(session, telegram_id)

    c = await add_commitment(
        session,
        user_id=user.id,
        peer_id=0,  # self-reminder
        peer_name="self",
        message_id=None,
        direction="mine",
        text=text,
        deadline_at=deadline_at,
    )

    return {
        "ok": True,
        "id": c.id,
        "text": c.text,
        "deadline": c.deadline_at.isoformat() if c.deadline_at else None,
    }
