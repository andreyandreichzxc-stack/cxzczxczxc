"""Cross-chat search tool — registered via @tool decorator.

Wraps ``src.db.repo.cross_chat_search`` as a tool that the LLM can invoke
to find conversations matching a text query across all chats.
"""

from __future__ import annotations

import logging
from typing import Any

from src.core.actions.tool_registry import tool
from src.db.repo import cross_chat_search

logger = logging.getLogger(__name__)


@tool(
    name="cross_chat_search",
    description=(
        "Search ALL conversations for a text query. "
        "Returns the top matching chats with highlighted message snippets "
        "so you can quickly see who discussed what topic."
    ),
    category="search",
    risk="low",
    params={
        "query": "str",
        "limit": "int",
    },
)
async def _cross_search_tool(
    query: str,
    limit: int = 5,
    **kwargs: Any,
) -> dict[str, Any]:
    """Search across all chats, return top conversations with snippets.

    Expects ``session`` and ``user`` in *kwargs* (injected by the caller).

    Returns:
        A dict with ``"ok": True`` and ``"results": [...]``, or
        ``{"error": "..."}`` on failure.
    """
    session = kwargs.get("session")
    user = kwargs.get("user")

    if session is None or user is None:
        return {"error": "Missing required runtime dependencies: session, user"}

    try:
        results = await cross_chat_search(session, user, query, limit=limit)
        return {
            "ok": True,
            "query": query,
            "results": results,
            "count": len(results),
        }
    except Exception:
        logger.exception("cross_chat_search failed")
        return {"error": "Search failed, please try again later"}
