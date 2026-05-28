"""Session search tool — FTS5 search across agent-owner conversations."""

from __future__ import annotations

import logging
from typing import Any

from src.core.actions.tool_registry import tool
from src.config import settings
from src.db.session import get_session
from src.db.repo import get_or_create_user
from src.db.repos.session_repo import search_session_messages

logger = logging.getLogger(__name__)


@tool(
    name="search_sessions",
    description=(
        "Search past conversations between you and the bot (agent sessions). "
        "Finds messages matching a text query across all your sessions."
    ),
    category="search",
    risk="low",
    params={
        "query": "str",
        "limit": "int=5",
    },
)
async def _search_sessions_tool(
    query: str = "",
    limit: int = 5,
    **kwargs: Any,
) -> dict[str, Any]:
    """Search across all agent-owner session messages for a text query.

    Expects ``session`` and ``user`` in *kwargs* (injected by the caller),
    falling back to ``settings.owner_telegram_id`` when user is not provided.

    Returns:
        A dict with ``"ok": True`` and ``"results": [...]``, or
        ``{"error": "..."}`` on failure.
    """
    if not query.strip():
        return {"error": "query is required"}

    session = kwargs.get("session")
    if session is None:
        try:
            async with get_session() as session:
                owner = await get_or_create_user(
                    session,
                    kwargs.get("user", settings.owner_telegram_id),
                )
                results = await search_session_messages(session, owner, query, limit)
                return {"ok": True, "results": results, "count": len(results)}
        except Exception:
            logger.exception("search_sessions failed")
            return {"error": "Search failed, please try again later"}

    try:
        owner = await get_or_create_user(
            session,
            kwargs.get("user", settings.owner_telegram_id),
        )
        results = await search_session_messages(session, owner, query, limit)
        return {"ok": True, "results": results, "count": len(results)}
    except Exception:
        logger.exception("search_sessions failed")
        return {"error": "Search failed, please try again later"}
