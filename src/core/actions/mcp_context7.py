"""mcp_context7 tool — registered via @tool decorator.

Context7 Docs Search — search documentation for programming libraries
via the context7.com API.

Actions:
  - **resolve** — find library IDs matching a library name (e.g. "next.js").
  - **query** — search documentation for a query within a specific library.

Requires: ``CONTEXT7_API_KEY`` in .env (or settings.context7_api_key).

Examples::

    action="resolve" name="next.js"
    action="query" library="/vercel/next.js" query="middleware" limit=3
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

from src.config import settings
from src.core.actions.tool_registry import tool

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

_API_BASE = "https://api.context7.com/v1"
_RESOLVE_CACHE_TTL = 3600  # 1 hour
_QUERY_CACHE_TTL = 600  # 10 minutes

# ── In-memory caches ─────────────────────────────────────────────────────

_RESOLVE_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_QUERY_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}


# ══════════════════════════════════════════════════════════════════════════
# Tool: mcp_context7
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="mcp_context7",
    description=(
        "Search documentation for programming libraries via the Context7 API.\n\n"
        "Actions:\n"
        "- **resolve** — find library IDs by name (e.g. 'next.js').\n"
        "  Returns: list of matching libraries with id, name, description, "
        "snippets_count.\n"
        "- **query** — search documentation within a specific library.\n"
        "  Returns: list of code snippets with description and source URL.\n\n"
        "Requires: CONTEXT7_API_KEY set in .env\n\n"
        "Examples:\n"
        '  action="resolve" name="next.js"\n'
        '  action="query" library="/vercel/next.js" query="middleware" limit=3'
    ),
    category="search",
    risk="low",
    params={
        "action": "str — 'resolve' or 'query'",
        "name": "str — library name to resolve (required for resolve)",
        "library": "str — library ID like '/vercel/next.js' (required for query)",
        "query": "str — search query (required for query)",
        "limit": "int — max results (default 5, max 20)",
    },
)
async def mcp_context7(
    action: str = "",
    name: str = "",
    library: str = "",
    query: str = "",
    limit: int = 5,
    **kwargs: Any,
) -> dict[str, Any]:
    """Context7 docs search tool — resolve libraries and query documentation."""
    try:
        api_key = settings.context7_api_key
        if not api_key:
            return {
                "error": "Set CONTEXT7_API_KEY in .env — get your key at https://context7.com"
            }

        if action == "resolve":
            if not name:
                return {"error": "name parameter is required for action='resolve'"}
            return await _resolve_library(api_key, name.strip())

        elif action == "query":
            if not library:
                return {"error": "library parameter is required for action='query'"}
            if not query:
                return {"error": "query parameter is required for action='query'"}
            if limit < 1:
                limit = 1
            elif limit > 20:
                limit = 20
            return await _query_docs(api_key, library.strip(), query.strip(), limit)

        else:
            return {
                "error": (f"Unknown action {action!r}. Valid actions: resolve, query")
            }

    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        logger.exception("mcp_context7(%r) failed", action)
        return {"error": f"Unexpected error: {exc}"}


# ══════════════════════════════════════════════════════════════════════════
# API Helpers
# ══════════════════════════════════════════════════════════════════════════


async def _resolve_library(api_key: str, name: str) -> dict[str, Any]:
    """Resolve a library name to library IDs via ``/v1/resolve-library``.

    Results are cached in-memory with a 1-hour TTL keyed by the library name.
    """
    now = time.monotonic()

    # ── Check cache ──────────────────────────────────────────────────
    cached = _RESOLVE_CACHE.get(name)
    if cached and (now - cached[0]) < _RESOLVE_CACHE_TTL:
        return {"ok": True, "action": "resolve", "results": cached[1]}

    # ── API call ─────────────────────────────────────────────────────
    import httpx  # lazy import

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{_API_BASE}/resolve-library",
            headers=headers,
            json={"name": name},
        )

    if response.status_code == 401:
        return {"error": "Invalid CONTEXT7_API_KEY — check your .env"}
    if response.status_code != 200:
        return {
            "error": (
                f"Context7 API error: HTTP {response.status_code} — {response.text}"
            )
        }

    data = response.json()
    raw_items = data if isinstance(data, list) else data.get("results", [])

    results: list[dict[str, Any]] = []
    for item in raw_items:
        results.append(
            {
                "id": item.get("id") or item.get("libraryId", ""),
                "name": item.get("name", ""),
                "description": item.get("description", ""),
                "snippets_count": item.get("snippets_count")
                or item.get("snippetCount", 0),
            }
        )

    # ── Update cache ─────────────────────────────────────────────────
    _RESOLVE_CACHE[name] = (now, results)

    return {"ok": True, "action": "resolve", "results": results}


async def _query_docs(
    api_key: str,
    library_id: str,
    query_str: str,
    limit: int,
) -> dict[str, Any]:
    """Search documentation via ``/v1/query-docs``.

    Results are cached in-memory with a 10-minute TTL.
    Cache key is ``sha256(library_id + "|" + query)``.
    """
    # ── Build cache key ──────────────────────────────────────────────
    raw_key = f"{library_id}|{query_str}"
    cache_key = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    now = time.monotonic()

    # ── Check cache ──────────────────────────────────────────────────
    cached = _QUERY_CACHE.get(cache_key)
    if cached and (now - cached[0]) < _QUERY_CACHE_TTL:
        return {"ok": True, "action": "query", "results": cached[1][:limit]}

    # ── API call ─────────────────────────────────────────────────────
    import httpx  # lazy import

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{_API_BASE}/query-docs",
            headers=headers,
            json={
                "libraryId": library_id,
                "query": query_str,
            },
        )

    if response.status_code == 401:
        return {"error": "Invalid CONTEXT7_API_KEY — check your .env"}
    if response.status_code != 200:
        return {
            "error": (
                f"Context7 API error: HTTP {response.status_code} — {response.text}"
            )
        }

    data = response.json()
    raw_items = data if isinstance(data, list) else data.get("results", [])

    results: list[dict[str, Any]] = []
    for item in raw_items:
        results.append(
            {
                "code": item.get("code", ""),
                "description": item.get("description", ""),
                "source": item.get("source", ""),
            }
        )

    # ── Update cache (store full list, slice on retrieval) ───────────
    _QUERY_CACHE[cache_key] = (now, results)

    return {"ok": True, "action": "query", "results": results[:limit]}
