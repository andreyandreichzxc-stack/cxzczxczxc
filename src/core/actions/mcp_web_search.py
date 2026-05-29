"""MCP Tool: веб-поиск через DuckDuckGo."""

import asyncio
import logging
from typing import Any

from src.core.actions.tool_registry import tool

logger = logging.getLogger(__name__)


@tool(
    name="web_search",
    description="Ищет в интернете и возвращает сниппеты. Используй когда не знаешь ответа — сначала поищи!",
    category="web",
    risk="low",
    params={
        "query": "str — поисковый запрос",
        "limit": "int — макс. результатов (1-10, по умолчанию 3)",
    },
)
async def web_search(
    query: str = "",
    limit: int = 3,
    **kwargs: Any,
) -> dict[str, Any]:
    if not query:
        return {"error": "query обязателен"}
    limit = max(1, min(10, limit))

    try:
        from duckduckgo_search import DDGS

        def _search():
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=limit))

        results = await asyncio.wait_for(asyncio.to_thread(_search), timeout=15.0)

        if not results:
            return {"ok": True, "results": [], "query": query}

        items = []
        for r in results:
            items.append(
                {
                    "title": r.get("title", ""),
                    "snippet": r.get("body", "")[:300],
                    "url": r.get("href", ""),
                }
            )

        return {"ok": True, "results": items, "query": query}

    except ImportError:
        return {
            "error": "duckduckgo-search не установлен. pip install duckduckgo-search"
        }
    except Exception as e:
        return {"error": str(e)[:300]}
