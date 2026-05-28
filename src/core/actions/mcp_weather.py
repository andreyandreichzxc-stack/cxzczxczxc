"""mcp_weather tool — registered via @tool decorator.

Fetches weather data from wttr.in (free, no API key required).

Features:
- ``action="current"`` — short text format (``?format=3``).
- ``action="forecast"`` — full JSON forecast with configurable days.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import requests

from src.core.actions.tool_registry import tool

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

_WTTR_BASE = "https://wttr.in"
_FETCH_TIMEOUT = 10  # seconds
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

_REQUEST_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "text/plain, application/json",
}


# ══════════════════════════════════════════════════════════════════════════
# Tool: mcp_weather
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="mcp_weather",
    description=(
        "Get weather data for a city from wttr.in.  Supports two actions:\n"
        "- 'current' — return a short text summary of current weather.\n"
        "- 'forecast' — return a full JSON forecast for the specified number "
        "of days (1–7)."
    ),
    category="search",
    risk="low",
    params={
        "action": "str — 'current' or 'forecast'",
        "city": "str — city name (e.g. 'Moscow', 'London')",
        "days": "int — number of forecast days (1-7, default 3, used with forecast)",
    },
)
async def mcp_weather(
    action: str,
    city: str = "",
    days: int = 3,
    **kwargs: Any,
) -> dict[str, Any]:
    """Weather data tool.

    Args:
        action: ``"current"`` or ``"forecast"``.
        city: City name (e.g. ``"Moscow"``).
        days: Forecast days (1–7, default 3, used with ``action="forecast"``).

    Returns:
        A dict with weather data or an ``"error"`` key on failure.
    """
    try:
        if action not in ("current", "forecast"):
            return {
                "error": f"Unknown action {action!r}. Valid actions: current, forecast"
            }

        if not city or not city.strip():
            return {"error": "city parameter is required"}

        city = city.strip()

        if action == "current":
            return await _current_weather(city)
        else:
            days = max(1, min(days, 7))
            return await _forecast_weather(city, days)
    except Exception as exc:
        logger.exception("mcp_weather(%r) failed", action)
        return {"error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════
# Action implementations
# ══════════════════════════════════════════════════════════════════════════


async def _current_weather(city: str) -> dict[str, Any]:
    """Fetch short current-weather text for *city*."""

    loop = asyncio.get_running_loop()

    def _fetch() -> str:
        url = f"{_WTTR_BASE}/{city}?format=3"
        try:
            resp = requests.get(url, headers=_REQUEST_HEADERS, timeout=_FETCH_TIMEOUT)
            resp.raise_for_status()
            return resp.text.strip()
        except requests.RequestException as exc:
            logger.warning("Failed to fetch weather for %r: %s", city, exc)
            raise

    try:
        text = await loop.run_in_executor(None, _fetch)
    except requests.RequestException as exc:
        return {"error": f"Weather request failed: {exc}"}

    return {
        "ok": True,
        "city": city,
        "weather": text,
    }


async def _forecast_weather(city: str, days: int) -> dict[str, Any]:
    """Fetch full JSON forecast for *city* for *days* days."""

    loop = asyncio.get_running_loop()

    def _fetch() -> dict[str, Any]:
        url = f"{_WTTR_BASE}/{city}?format=j1"
        try:
            resp = requests.get(url, headers=_REQUEST_HEADERS, timeout=_FETCH_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            logger.warning("Failed to fetch forecast for %r: %s", city, exc)
            raise
        except json.JSONDecodeError as exc:
            logger.warning("Invalid JSON from wttr.in for %r: %s", city, exc)
            raise ValueError(f"Invalid response from weather service: {exc}") from exc

    try:
        data = await loop.run_in_executor(None, _fetch)
    except requests.RequestException as exc:
        return {"error": f"Weather request failed: {exc}"}
    except ValueError as exc:
        return {"error": str(exc)}

    # Extract current conditions
    current = data.get("current_condition", [{}])[0]
    # Extract forecast days
    forecasts = data.get("weather", [])[:days]

    simplified_forecast = []
    for day in forecasts:
        simplified_forecast.append(
            {
                "date": day.get("date", ""),
                "max_temp_c": day.get("maxtempC", ""),
                "min_temp_c": day.get("mintempC", ""),
                "description": (
                    day.get("hourly", [{}])[0]
                    .get("weatherDesc", [{}])[0]
                    .get("value", "")
                ),
            }
        )

    return {
        "ok": True,
        "city": city,
        "current_temp_c": current.get("temp_C", ""),
        "current_desc": (current.get("weatherDesc", [{}])[0].get("value", "")),
        "humidity": current.get("humidity", ""),
        "wind_speed_kmh": current.get("windspeedKmph", ""),
        "forecast": simplified_forecast,
        "forecast_days": len(simplified_forecast),
    }
