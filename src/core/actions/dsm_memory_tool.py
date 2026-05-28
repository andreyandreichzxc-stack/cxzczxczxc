"""dsm_memory tool — search project memory (DSM)."""

from __future__ import annotations

import logging
from typing import Any

from src.core.actions.tool_registry import tool
from src.core.intelligence.dsm import dsm_search, dsm_get_recent, dsm_list_tags

logger = logging.getLogger(__name__)


@tool(
    name="dsm_memory",
    description=(
        "Поиск по проектной памяти (DSM) — что обсуждали, какие решения приняли, "
        "какие баги фиксили. Используй когда нужно вспомнить архитектурные решения, "
        "дизайн-выборы, договорённости по коду. "
        "Поддерживает action='search', 'recent' и 'tags'."
    ),
    category="memory",
    risk="low",
    params={
        "action": "str — 'search' (поиск), 'recent' (последние записи), 'tags' (список тегов)",
        "query": "str — поисковый запрос (для action='search')",
        "limit": "int=5 — макс. результатов",
    },
)
async def _dsm_memory_tool(
    action: str = "search",
    query: str = "",
    limit: int = 5,
    **kwargs: Any,
) -> dict[str, Any]:
    """Search DSM project memory.

    Returns:
        For ``search``: ``{"ok": True, "results": [...], "count": N}``
        For ``recent``: ``{"ok": True, "results": [...], "count": N}``
        For ``tags``: ``{"ok": True, "tags": [...], "count": N}``
    """
    if action == "search":
        results = await dsm_search(query, limit)
        return {"ok": True, "results": results, "count": len(results)}
    elif action == "recent":
        results = await dsm_get_recent(limit=limit)
        return {"ok": True, "results": results, "count": len(results)}
    elif action == "tags":
        tags = await dsm_list_tags()
        return {"ok": True, "tags": tags, "count": len(tags)}
    return {"error": f"Unknown action: {action!r}"}
