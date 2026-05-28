"""mcp_monitor tool — registered via @tool decorator.

System resource monitoring via psutil.

Actions:
- ``action="cpu"`` — CPU usage per-core + overall, CPU count, load average
- ``action="memory"`` — total, available, used, percent (human-readable GB)
- ``action="disk" path="/"`` — total, used, free, percent for a given path
- ``action="uptime"`` — system uptime + bot process uptime
- ``action="all"`` — all of the above in one call
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from src.core.actions.tool_registry import tool
from src.config import settings

logger = logging.getLogger(__name__)

# ── Module-level: bot start time ─────────────────────────────────────────

_BOT_START_TIME: float = time.time()


# ══════════════════════════════════════════════════════════════════════════
# Tool: mcp_monitor
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="mcp_monitor",
    description=(
        "Monitor system resources (CPU, memory, disk, uptime). Supports five actions:\n"
        "- 'cpu' — CPU usage (per-core + overall), CPU count, load average.\n"
        "- 'memory' — RAM total, available, used, percent.\n"
        "- 'disk' — disk usage for a given path (default: data/).\n"
        "- 'uptime' — system uptime and bot process uptime.\n"
        "- 'all' — all of the above in one call."
    ),
    category="system",
    risk="low",
    params={
        "action": "str — 'cpu', 'memory', 'disk', 'uptime', or 'all'",
        "path": "str — disk path to check (default: data/, used with 'disk')",
    },
)
async def mcp_monitor(
    action: str,
    path: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """System resource monitoring tool.

    Args:
        action: ``"cpu"``, ``"memory"``, ``"disk"``, ``"uptime"``, or ``"all"``.
        path: Disk path to check (default: data directory, used with ``action="disk"``).

    Returns:
        A dict with the requested resource data or an ``"error"`` key on failure.
    """
    try:
        if action == "cpu":
            return await _monitor_cpu()
        elif action == "memory":
            return await _monitor_memory()
        elif action == "disk":
            return await _monitor_disk(path or str(settings.data_dir))
        elif action == "uptime":
            return await _monitor_uptime()
        elif action == "all":
            return await _monitor_all()
        else:
            return {
                "error": (
                    f"Unknown action {action!r}. "
                    "Valid actions: cpu, memory, disk, uptime, all"
                )
            }
    except Exception as exc:
        logger.exception("mcp_monitor(%r) failed", action)
        return {"error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════
# psutil lazy import helper
# ══════════════════════════════════════════════════════════════════════════


def _get_psutil() -> Any:
    """Lazy import of psutil. Returns the module or raises ImportError."""
    try:
        import psutil  # type: ignore[import-untyped]

        return psutil
    except ImportError:
        raise ImportError("psutil not installed: pip install psutil")


# ══════════════════════════════════════════════════════════════════════════
# Action implementations
# ══════════════════════════════════════════════════════════════════════════


async def _monitor_cpu() -> dict[str, Any]:
    """Collect CPU information."""
    psutil = _get_psutil()
    loop = asyncio.get_running_loop()

    def _collect() -> dict[str, Any]:
        per_core = psutil.cpu_percent(percpu=True, interval=0.1)
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "cpu_percent_per_core": per_core,
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "load_avg": [round(x, 2) for x in psutil.getloadavg()],
        }

    try:
        data = await loop.run_in_executor(None, _collect)
    except ImportError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        logger.warning("CPU monitoring error: %s", exc)
        return {"error": f"CPU data collection failed: {exc}"}

    return {"ok": True, **data}


async def _monitor_memory() -> dict[str, Any]:
    """Collect memory information."""
    psutil = _get_psutil()
    loop = asyncio.get_running_loop()

    def _collect() -> dict[str, Any]:
        mem = psutil.virtual_memory()
        return {
            "total_gb": round(mem.total / (1024**3), 2),
            "available_gb": round(mem.available / (1024**3), 2),
            "used_gb": round(mem.used / (1024**3), 2),
            "percent": mem.percent,
        }

    try:
        data = await loop.run_in_executor(None, _collect)
    except ImportError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        logger.warning("Memory monitoring error: %s", exc)
        return {"error": f"Memory data collection failed: {exc}"}

    return {"ok": True, **data}


async def _monitor_disk(disk_path: str) -> dict[str, Any]:
    """Collect disk usage for *disk_path*."""
    psutil = _get_psutil()
    loop = asyncio.get_running_loop()

    def _collect() -> dict[str, Any]:
        usage = psutil.disk_usage(disk_path)
        return {
            "path": disk_path,
            "total_gb": round(usage.total / (1024**3), 2),
            "used_gb": round(usage.used / (1024**3), 2),
            "free_gb": round(usage.free / (1024**3), 2),
            "percent": usage.percent,
        }

    try:
        data = await loop.run_in_executor(None, _collect)
    except ImportError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        logger.warning("Disk monitoring error: %s", exc)
        return {"error": f"Disk data collection failed: {exc}"}

    return {"ok": True, **data}


async def _monitor_uptime() -> dict[str, Any]:
    """Collect system and bot uptime."""
    psutil = _get_psutil()
    loop = asyncio.get_running_loop()

    def _collect() -> dict[str, Any]:

        boot_ts = psutil.boot_time()
        now = time.time()
        system_uptime_sec = now - boot_ts
        bot_uptime_sec = now - _BOT_START_TIME

        return {
            "system_uptime_sec": round(system_uptime_sec),
            "system_uptime_human": _format_uptime(system_uptime_sec),
            "bot_uptime_sec": round(bot_uptime_sec),
            "bot_uptime_human": _format_uptime(bot_uptime_sec),
            "boot_time_iso": _timestamp_iso(boot_ts),
        }

    try:
        data = await loop.run_in_executor(None, _collect)
    except ImportError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        logger.warning("Uptime monitoring error: %s", exc)
        return {"error": f"Uptime data collection failed: {exc}"}

    return {"ok": True, **data}


async def _monitor_all() -> dict[str, Any]:
    """Collect all metrics in one call."""
    cpu_result = await _monitor_cpu()
    if "error" in cpu_result:
        return cpu_result

    mem_result = await _monitor_memory()
    if "error" in mem_result:
        return mem_result

    disk_result = await _monitor_disk(str(settings.data_dir))
    uptime_result = await _monitor_uptime()

    # Merge — each result has "ok": True prefix, we strip it
    merged: dict[str, Any] = {"ok": True}
    for section_name, section_data in [
        ("cpu", cpu_result),
        ("memory", mem_result),
        ("disk", disk_result),
        ("uptime", uptime_result),
    ]:
        merged[section_name] = {k: v for k, v in section_data.items() if k != "ok"}

    return merged


# ── Helpers ────────────────────────────────────────────────────────────


def _format_uptime(seconds: float) -> str:
    """Format seconds to a human-readable string like '2d 3h 15m 42s'."""
    days, rem = divmod(int(seconds), 86_400)
    hours, rem = divmod(rem, 3_600)
    minutes, secs = divmod(rem, 60)

    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def _timestamp_iso(ts: float) -> str:
    """Convert Unix timestamp to ISO 8601 string."""
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
