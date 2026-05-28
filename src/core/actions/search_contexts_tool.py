"""search_contexts tool — registered via @tool decorator.

Wraps ``src.core.memory.context_files`` as a tool that the LLM can invoke to search
the user's context notes (owner profile, contact profiles, arbitrary
knowledge files stored under ``data/contexts/``).

Supports hybrid search: FTS5 (keywords) + Qdrant (semantic) via RRF.
"""

from __future__ import annotations

import logging
from typing import Any

from src.core.actions.tool_registry import tool
from src.core.memory.context_files import (
    list_context_files,
    search_contexts_hybrid,
    search_in_contexts,
)

logger = logging.getLogger(__name__)


@tool(
    name="search_contexts",
    description=(
        "Поиск по контекстным заметкам (о владельце и контактах). "
        "Используй когда нужно найти информацию о принципах, предпочтениях, "
        "фактах о пользователе или его контактах. "
        "Гибридный поиск: семантический (по смыслу) + ключевые слова. "
        "Поддерживает action='search' и action='list'."
    ),
    category="search",
    risk="low",
    params={
        "action": "str — 'search' для поиска или 'list' для списка всех ключей",
        "query": "str — поисковый запрос (только для action='search')",
        "limit": "int=5 — максимум результатов (только для action='search')",
    },
    output_schema={
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "filename without .md",
                        },
                        "snippet": {
                            "type": "string",
                            "description": "Search result with <b>highlighting</b>",
                        },
                        "score": {"type": "number", "description": "Relevance score"},
                    },
                },
            },
            "count": {"type": "integer"},
        },
        "required": ["ok"],
    },
)
async def _search_contexts_tool(
    action: str = "search",
    query: str = "",
    limit: int = 5,
    **kwargs: Any,
) -> dict[str, Any]:
    """Search or list context files.

    ``action="search"`` — hybrid search (FTS5 keywords + Qdrant semantic)
    across all .md context files with ranked results and highlighted snippets.

    ``action="list"`` — return all context file keys (without .md extension).

    Returns:
        For ``search``: ``{"ok": True, "results": [{"key": ..., "snippet": ..., "score": ...}], "count": N}``
        For ``list``: ``{"ok": True, "keys": [...], "count": N}``
    """
    if action == "list":
        keys = list_context_files()
        return {"ok": True, "keys": keys, "count": len(keys)}

    if action == "search":
        if not query.strip():
            return {"error": "query is required for action='search'"}
        try:
            # Try hybrid search (FTS5 + semantic) if provider is available
            provider = kwargs.get("provider")
            if provider:
                results = await search_contexts_hybrid(
                    query, provider=provider, limit=limit
                )
            else:
                results = search_in_contexts(query, limit=limit)
            return {"ok": True, "results": results, "count": len(results)}
        except Exception:
            logger.exception("search_contexts tool failed")
            return {"error": "Search failed, please try again later"}

    return {"error": f"Unknown action: {action!r}. Use 'search' or 'list'."}
