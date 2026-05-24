"""search_contexts tool — registered via @tool decorator.

Wraps ``src.core.memory.context_files.search_in_contexts`` and
``list_context_files`` as a tool that the LLM can invoke to search
the user's context notes (owner profile, contact profiles, arbitrary
knowledge files stored under ``data/contexts/``).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.core.actions.tool_registry import tool
from src.core.memory.context_files import list_context_files, search_in_contexts

logger = logging.getLogger(__name__)


@tool(
    name="search_contexts",
    description=(
        "Поиск по контекстным заметкам (о владельце и контактах). "
        "Используй когда нужно найти информацию о принципах, предпочтениях, "
        "фактах о пользователе или его контактах. "
        "Поддерживает action='search' (поиск по запросу) и action='list' (список всех ключей)."
    ),
    category="search",
    risk="low",
    params={
        "action": "str — 'search' для поиска или 'list' для списка всех ключей",
        "query": "str — поисковый запрос (только для action='search')",
        "limit": "int=5 — максимум результатов (только для action='search')",
    },
)
async def _search_contexts_tool(
    action: str = "search",
    query: str = "",
    limit: int = 5,
    **kwargs: Any,
) -> dict[str, Any]:
    """Search or list context files.

    ``action="search"`` — full-text search across all .md context files
    with ranked results and highlighted snippets.

    ``action="list"`` — return all context file keys (without .md extension).

    Returns:
        For ``search``: ``{"ok": True, "results": [{"key": ..., "snippet": ..., "rank": ...}], "count": N}``
        For ``list``: ``{"ok": True, "keys": [...], "count": N}``
    """
    if action == "list":
        keys = list_context_files()
        return {"ok": True, "keys": keys, "count": len(keys)}

    if action == "search":
        if not query.strip():
            return {"error": "query is required for action='search'"}
        try:
            results = await asyncio.to_thread(search_in_contexts, query, limit)
            return {"ok": True, "results": results, "count": len(results)}
        except Exception:
            logger.exception("search_contexts tool failed")
            return {"error": "Search failed, please try again later"}

    return {"error": f"Unknown action: {action!r}. Use 'search' or 'list'."}
