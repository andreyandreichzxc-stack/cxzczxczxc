"""mcp_avito tool — registered via @tool decorator.

Search Avito listings and get market statistics.

Actions:
- ``action="search"`` — search listings with optional price/city filters.
- ``action="stats"`` — market statistics (avg, min, max price, counts).

Uses ``scan_avito()`` from ``src.core.avito.service``.
"""

from __future__ import annotations

import logging
from typing import Any

from src.core.actions.tool_registry import tool
from src.core.avito.service import SearchParams, scan_avito
from src.config import settings

logger = logging.getLogger(__name__)


@tool(
    name="mcp_avito",
    description=(
        "Search Avito listings and get market statistics. Supports two actions:\n"
        "- 'search' — search for listings with optional filters (query, city, "
        "price range). Returns listings with deal_score and scam check.\n"
        "- 'stats' — get market statistics for a query (avg/min/max price, "
        "count, new/used count)."
    ),
    category="search",
    risk="low",
    params={
        "action": "str — 'search' or 'stats'",
        "query": "str — search query (e.g. 'iphone', 'macbook')",
        "city": "str|None — city name (default: settings default city)",
        "max_price": "int|None — maximum price filter",
        "min_price": "int|None — minimum price filter",
        "limit": "int — max results to return (default 10, used with 'search')",
    },
)
async def mcp_avito(
    action: str,
    query: str = "",
    city: str | None = None,
    max_price: int | None = None,
    min_price: int | None = None,
    limit: int = 10,
    **kwargs: Any,
) -> dict[str, Any]:
    """Avito search and statistics tool.

    Args:
        action: ``"search"`` or ``"stats"``.
        query: Search query (e.g. ``"iphone"``).
        city: City name (defaults to ``settings.avito_default_city``).
        max_price: Maximum price filter.
        min_price: Minimum price filter.
        limit: Max results to return (default 10, only for ``action="search"``).

    Returns:
        A dict with result data or an ``"error"`` key on failure.
    """
    try:
        if action not in ("search", "stats"):
            return {
                "error": f"Unknown action {action!r}. Valid actions: search, stats",
            }

        if not query or not query.strip():
            return {"error": "query parameter is required"}

        query = query.strip()
        city = city or settings.avito_default_city
        limit = max(1, min(limit, 100))

        params = SearchParams(
            city=city,
            category="",
            query=query,
            price_min=min_price,
            price_max=max_price,
        )

        result = await scan_avito(params)

        if result.error:
            return {"error": result.error}

        if action == "search":
            return _format_search_result(result, limit)
        else:
            return _format_stats_result(result)

    except Exception as exc:
        logger.exception("mcp_avito(%r) failed", action)
        return {"error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════
# Formatting helpers
# ══════════════════════════════════════════════════════════════════════════


def _format_search_result(result: Any, limit: int) -> dict[str, Any]:
    """Format scan result for the 'search' action."""
    listings = []
    for item in result.listings[:limit]:
        # Normalise nested deal_score / scam_check for cleaner output
        deal = item.get("deal_score")
        if isinstance(deal, dict):
            deal_score_value = deal.get("score")
            deal_grade = deal.get("grade")
        else:
            deal_score_value = deal
            deal_grade = None

        scam = item.get("scam_check")
        if isinstance(scam, dict):
            is_suspicious = scam.get("is_suspicious", False)
            scam_risk = scam.get("risk", "unknown")
            scam_reasons = scam.get("reasons", [])
        else:
            is_suspicious = False
            scam_risk = "unknown"
            scam_reasons = []

        listing = {
            "avito_id": item.get("avito_id"),
            "title": item.get("title"),
            "price": item.get("price"),
            "url": item.get("url"),
            "image_url": item.get("image_url"),
            "city": item.get("city"),
            "condition": item.get("condition"),
            "delivery": item.get("delivery", False),
            "seller_name": item.get("seller_name"),
            "seller_rating": item.get("seller_rating"),
            "seller_reviews": item.get("seller_reviews"),
            "description": item.get("description"),
            "deal_score": deal_score_value,
            "deal_grade": deal_grade,
            "is_suspicious": is_suspicious,
            "scam_risk": scam_risk,
            "scam_reasons": scam_reasons,
        }
        listings.append(listing)

    return {
        "ok": True,
        "listings": listings,
        "count": len(listings),
        "total_found": result.total_parsed,
        "url": result.url,
    }


def _format_stats_result(result: Any) -> dict[str, Any]:
    """Format scan result for the 'stats' action."""
    prices = [
        item["price"] for item in result.listings if item.get("price") is not None
    ]

    new_count = sum(1 for item in result.listings if item.get("condition") == "new")
    used_count = sum(
        1
        for item in result.listings
        if item.get("condition") is not None and item["condition"] != "new"
    )

    if not prices:
        return {
            "ok": True,
            "avg_price": None,
            "min_price": None,
            "max_price": None,
            "count": 0,
            "new_count": 0,
            "used_count": 0,
            "url": result.url,
        }

    return {
        "ok": True,
        "avg_price": round(sum(prices) / len(prices), 2),
        "min_price": float(min(prices)),
        "max_price": float(max(prices)),
        "count": len(prices),
        "new_count": new_count,
        "used_count": used_count,
        "url": result.url,
    }
