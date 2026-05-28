"""mcp_crypto tool — registered via @tool decorator.

Fetches cryptocurrency prices from the CoinGecko public API (free, rate-limited).

Features:
- ``action="price"`` — get current price for a coin in a given currency.
- ``action="top"`` — list top N coins by market cap.
- Handles HTTP 429 (rate limit) gracefully with a retry hint.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import requests

from src.core.actions.tool_registry import tool

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

_COINGECKO_BASE = "https://api.coingecko.com/api/v3"
_FETCH_TIMEOUT = 10  # seconds
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

_REQUEST_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "application/json",
}


# ══════════════════════════════════════════════════════════════════════════
# Tool: mcp_crypto
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="mcp_crypto",
    description=(
        "Get cryptocurrency prices and market data from CoinGecko.  Supports "
        "two actions:\n"
        "- 'price' — get the current price of a coin (e.g. 'bitcoin') in the "
        "given currency (e.g. 'usd').\n"
        "- 'top' — list the top N coins by market capitalisation."
    ),
    category="search",
    risk="low",
    params={
        "action": "str — 'price' or 'top'",
        "coin": "str — coin ID (e.g. 'bitcoin', 'ethereum'), required for price",
        "currency": "str — vs currency (e.g. 'usd', 'eur'), default 'usd'",
        "limit": "int — number of top coins (1-50, default 5), used with top",
    },
)
async def mcp_crypto(
    action: str,
    coin: str = "",
    currency: str = "usd",
    limit: int = 5,
    **kwargs: Any,
) -> dict[str, Any]:
    """Cryptocurrency price tool.

    Args:
        action: ``"price"`` or ``"top"``.
        coin: Coin ID (e.g. ``"bitcoin"``, required for ``action="price"``).
        currency: Vs currency (e.g. ``"usd"``, default ``"usd"``).
        limit: Number of top coins (1–50, default 5, used with ``action="top"``).

    Returns:
        A dict with price data or an ``"error"`` key on failure.
    """
    try:
        if action not in ("price", "top"):
            return {"error": f"Unknown action {action!r}. Valid actions: price, top"}

        if action == "price":
            if not coin or not coin.strip():
                return {"error": "coin parameter is required for action='price'"}
            return await _get_price(coin.strip().lower(), currency.strip().lower())
        else:
            limit = max(1, min(limit, 50))
            return await _get_top(limit)
    except Exception as exc:
        logger.exception("mcp_crypto(%r) failed", action)
        return {"error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════
# Action implementations
# ══════════════════════════════════════════════════════════════════════════


async def _get_price(coin: str, currency: str) -> dict[str, Any]:
    """Fetch current price for *coin* in *currency*."""

    loop = asyncio.get_running_loop()

    def _fetch() -> dict[str, Any]:
        url = f"{_COINGECKO_BASE}/simple/price?ids={coin}&vs_currencies={currency}"
        try:
            resp = requests.get(url, headers=_REQUEST_HEADERS, timeout=_FETCH_TIMEOUT)
        except requests.RequestException as exc:
            logger.warning("CoinGecko request failed for %r: %s", coin, exc)
            raise

        if resp.status_code == 429:
            raise _RateLimitError(
                "CoinGecko API rate limit hit. Please wait ~30s before retrying."
            )
        resp.raise_for_status()
        return resp.json()

    try:
        data = await loop.run_in_executor(None, _fetch)
    except _RateLimitError as exc:
        return {"error": str(exc)}
    except requests.RequestException as exc:
        return {"error": f"CoinGecko request failed: {exc}"}

    if coin not in data:
        return {
            "error": f"Coin '{coin}' not found. Check the coin ID (e.g. 'bitcoin', 'ethereum')."
        }

    price = data[coin].get(currency)
    if price is None:
        return {"error": f"Currency '{currency}' not available for {coin}."}

    return {
        "ok": True,
        "coin": coin,
        "currency": currency,
        "price": price,
    }


async def _get_top(limit: int) -> dict[str, Any]:
    """Fetch top *limit* coins by market cap."""

    loop = asyncio.get_running_loop()

    def _fetch() -> list[dict[str, Any]]:
        url = (
            f"{_COINGECKO_BASE}/coins/markets"
            f"?vs_currency=usd&order=market_cap_desc"
            f"&per_page={limit}&page=1&sparkline=false"
        )
        try:
            resp = requests.get(url, headers=_REQUEST_HEADERS, timeout=_FETCH_TIMEOUT)
        except requests.RequestException as exc:
            logger.warning("CoinGecko markets request failed: %s", exc)
            raise

        if resp.status_code == 429:
            raise _RateLimitError(
                "CoinGecko API rate limit hit. Please wait ~30s before retrying."
            )
        resp.raise_for_status()
        return resp.json()

    try:
        data = await loop.run_in_executor(None, _fetch)
    except _RateLimitError as exc:
        return {"error": str(exc)}
    except requests.RequestException as exc:
        return {"error": f"CoinGecko request failed: {exc}"}

    coins = []
    for c in data:
        coins.append(
            {
                "rank": c.get("market_cap_rank"),
                "name": c.get("name"),
                "symbol": c.get("symbol", "").upper(),
                "price_usd": c.get("current_price"),
                "market_cap": c.get("market_cap"),
                "change_24h": c.get("price_change_percentage_24h"),
            }
        )

    return {
        "ok": True,
        "coins": coins,
        "count": len(coins),
    }


# ── Custom exception ─────────────────────────────────────────────────────


class _RateLimitError(Exception):
    """Raised when the CoinGecko API returns HTTP 429."""
