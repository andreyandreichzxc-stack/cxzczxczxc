"""mcp_processes tool — registered via @tool decorator.

List and kill system processes via psutil.

Actions:
- ``action="list" sort_by="cpu" limit=10`` — top processes by CPU or memory
- ``action="find" name="python"`` — find processes matching a name
- ``action="kill" pid=1234`` — terminate a process (requires confirmation)

Output is sanitised: command-line arguments are stripped for privacy.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.core.actions.tool_registry import ToolActionSpec, tool

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

_MAX_PROCESSES = 50


# ══════════════════════════════════════════════════════════════════════════
# Tool: mcp_processes
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="mcp_processes",
    description=(
        "List and manage system processes.  Supports three actions:\n"
        "- 'list' — show top processes by CPU or memory usage.\n"
        "- 'find' — find processes by name substring.\n"
        "- 'kill' — terminate a process by PID (requires user confirmation).\n"
        "Command-line arguments are excluded from output for privacy."
    ),
    category="system",
    risk="medium",
    requires_confirmation=True,
    actions={
        "list": ToolActionSpec(name="list", risk="low", read_only=True, idempotent=True),
        "find": ToolActionSpec(name="find", risk="low", read_only=True, idempotent=True),
        "kill": ToolActionSpec(
            name="kill",
            risk="critical",
            read_only=False,
            destructive=True,
            idempotent=False,
            requires_confirmation=True,
        ),
    },
    params={
        "action": "str — 'list', 'find', or 'kill'",
        "sort_by": "str — 'cpu' or 'memory' (default 'cpu', used with action='list')",
        "limit": "int — max results (default 10, max 50, used with action='list'/'find')",
        "name": "str — process name substring (used with action='find')",
        "pid": "int — process ID to kill (required for action='kill')",
    },
)
async def mcp_processes(
    action: str,
    sort_by: str = "cpu",
    limit: int = 10,
    name: str = "",
    pid: int = 0,
    **kwargs: Any,
) -> dict[str, Any]:
    """System process management tool.

    Args:
        action: ``"list"``, ``"find"``, or ``"kill"``.
        sort_by: ``"cpu"`` or ``"memory"`` (default ``"cpu"``).
        limit: Max results (default 10, max 50).
        name: Process name substring (for ``action="find"``).
        pid: Process ID to kill (required for ``action="kill"``).

    Returns:
        A dict with process information or an ``"error"`` key.
    """
    try:
        if action not in ("list", "find", "kill"):
            return {
                "error": f"Unknown action {action!r}. Valid actions: list, find, kill"
            }

        if action == "list":
            n = max(1, min(limit, _MAX_PROCESSES))
            return await _list_processes(sort_by, n)
        elif action == "find":
            if not name or not name.strip():
                return {"error": "name parameter is required for action='find'"}
            n = max(1, min(limit, _MAX_PROCESSES))
            return await _find_processes(name.strip(), n)
        else:  # kill
            if pid <= 0:
                return {"error": "pid parameter must be a positive integer"}
            if not bool(kwargs.get("_confirmed", False)):
                return {
                    "error": "Action 'kill' requires confirmation",
                    "requires_confirmation": True,
                    "action": "kill",
                    "pid": pid,
                }
            return await _kill_process(pid)
    except Exception as exc:
        logger.exception("mcp_processes(%r) failed", action)
        return {"error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════
# Implementation
# ══════════════════════════════════════════════════════════════════════════


def _import_psutil() -> Any:
    """Lazy import of psutil. Raises ImportError if not installed."""
    try:
        import psutil  # type: ignore[import-untyped]

        return psutil
    except ImportError:
        raise ImportError("psutil not installed")


def _sanitize_proc(proc: Any) -> dict[str, Any]:
    """Extract a sanitised snapshot of a process (no cmdline args)."""
    try:
        pid = proc.info.get("pid") or proc.pid
    except Exception:
        pid = 0
    try:
        pname = proc.info.get("name") or proc.name()
    except Exception:
        pname = "?"
    try:
        cpu = proc.info.get("cpu_percent") or proc.cpu_percent(interval=None)
    except Exception:
        cpu = 0.0
    try:
        mem = proc.info.get("memory_percent") or proc.memory_percent()
    except Exception:
        mem = 0.0
    try:
        status = proc.info.get("status") or proc.status()
    except Exception:
        status = "?"

    return {
        "pid": pid,
        "name": pname,
        "cpu_percent": round(cpu, 1),
        "memory_percent": round(mem, 1),
        "status": status,
    }


async def _list_processes(sort_by: str, limit: int) -> dict[str, Any]:
    """Return top processes sorted by CPU or memory."""
    psutil = _import_psutil()
    loop = asyncio.get_running_loop()

    def _do() -> dict[str, Any]:
        processes: list[dict[str, Any]] = []
        for proc in psutil.process_iter(
            attrs=["pid", "name", "cpu_percent", "memory_percent", "status"]
        ):
            try:
                info = _sanitize_proc(proc)
                if (
                    info["cpu_percent"] is not None
                    or info["memory_percent"] is not None
                ):
                    processes.append(info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if sort_by == "memory":
            processes.sort(key=lambda p: p.get("memory_percent", 0) or 0, reverse=True)
        else:
            processes.sort(key=lambda p: p.get("cpu_percent", 0) or 0, reverse=True)

        return {"ok": True, "processes": processes[:limit], "total": len(processes)}

    try:
        return await loop.run_in_executor(None, _do)
    except ImportError as exc:
        return {"error": str(exc)}


async def _find_processes(name: str, limit: int) -> dict[str, Any]:
    """Find processes whose name contains *name* (case-insensitive)."""
    psutil = _import_psutil()
    loop = asyncio.get_running_loop()
    name_lower = name.lower()

    def _do() -> dict[str, Any]:
        matches: list[dict[str, Any]] = []
        for proc in psutil.process_iter(
            attrs=["pid", "name", "cpu_percent", "memory_percent", "status"]
        ):
            try:
                info = _sanitize_proc(proc)
                if name_lower in info["name"].lower():
                    matches.append(info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return {
            "ok": True,
            "query": name,
            "processes": matches[:limit],
            "count": len(matches),
        }

    try:
        return await loop.run_in_executor(None, _do)
    except ImportError as exc:
        return {"error": str(exc)}


async def _kill_process(pid: int) -> dict[str, Any]:
    """Terminate a process by PID.

    Note: the ``@tool`` decorator sets ``requires_confirmation=True``,
    so the caller must pass ``_confirmed=True`` for this to execute.
    """
    psutil = _import_psutil()
    loop = asyncio.get_running_loop()

    def _do() -> dict[str, Any]:
        try:
            proc = psutil.Process(pid)
            pname = proc.name()
            proc.terminate()
            return {
                "ok": True,
                "pid": pid,
                "name": pname,
                "message": f"Process {pid} ({pname}) terminated",
            }
        except psutil.NoSuchProcess:
            return {"error": f"Process {pid} not found"}
        except psutil.AccessDenied:
            return {"error": f"Access denied: cannot terminate process {pid}"}
        except Exception as exc:
            return {"error": f"Failed to kill process {pid}: {exc}"}

    try:
        return await loop.run_in_executor(None, _do)
    except ImportError as exc:
        return {"error": str(exc)}
