"""mcp_git tool — registered via @tool decorator.

Git operations via subprocess.  All commands are run in PROJECT_ROOT.

Actions:
- ``action="status"`` — ``git status --porcelain`` (changed files)
- ``action="log" count=5`` — ``git log --oneline -n {count}``
- ``action="diff" file=...`` — ``git diff`` for staged+unstaged on one file
- ``action="branch"`` — ``git branch --list`` (current branch first)
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from typing import Any

from src.config import PROJECT_ROOT
from src.core.actions.tool_registry import tool

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

_GIT_TIMEOUT = 10  # seconds


# ══════════════════════════════════════════════════════════════════════════
# Tool: mcp_git
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="mcp_git",
    description=(
        "Git operations on the project repository.  Supports four actions:\n"
        "- 'status' — show changed files (git status --porcelain).\n"
        "- 'log' — show recent commit log (git log --oneline).\n"
        "- 'diff' — show staged+unstaged diff for a specific file.\n"
        "- 'branch' — list branches (current branch first).\n"
        "All commands are run in the project root with a 10-second timeout."
    ),
    category="system",
    risk="low",
    params={
        "action": "str — 'status', 'log', 'diff', or 'branch'",
        "count": "int — number of log entries (default 5, used with action='log')",
        "file": "str — file path (required for action='diff')",
    },
)
async def mcp_git(
    action: str,
    count: int = 5,
    file: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """Git operations tool.

    Args:
        action: ``"status"``, ``"log"``, ``"diff"``, or ``"branch"``.
        count: Number of log entries (default 5, used with ``action="log"``).
        file: File path (required for ``action="diff"``).

    Returns:
        A dict with command output or an ``"error"`` key on failure.
    """
    try:
        if action not in ("status", "log", "diff", "branch"):
            return {
                "error": f"Unknown action {action!r}. "
                f"Valid actions: status, log, diff, branch"
            }

        if action == "status":
            return await _run_git(["status", "--porcelain"])
        elif action == "log":
            n = max(1, min(count, 100))
            return await _run_git(["log", "--oneline", f"-n{n}"])
        elif action == "diff":
            if not file or not file.strip():
                return {"error": "file parameter is required for action='diff'"}
            return await _run_git(["diff", "--", file.strip()])
        else:  # branch
            return await _run_git(["branch", "--list"])
    except Exception as exc:
        logger.exception("mcp_git(%r) failed", action)
        return {"error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════
# Implementation
# ══════════════════════════════════════════════════════════════════════════


async def _run_git(args: list[str]) -> dict[str, Any]:
    """Run a git command in PROJECT_ROOT with a 10-second timeout.

    Returns a dict with ``stdout``, ``stderr``, ``returncode``,
    or an ``"error"`` key on failure.
    """
    loop = asyncio.get_running_loop()

    def _do() -> dict[str, Any]:
        # Check that git is available
        try:
            subprocess.run(
                ["git", "--version"],
                capture_output=True,
                timeout=_GIT_TIMEOUT,
                text=True,
            )
        except FileNotFoundError:
            return {"error": "git not found"}
        except OSError:
            return {"error": "git not found"}

        # Check that we're in a git repo
        try:
            check = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                capture_output=True,
                timeout=_GIT_TIMEOUT,
                cwd=str(PROJECT_ROOT),
                text=True,
            )
        except subprocess.TimeoutExpired:
            return {"error": "git command timed out"}
        except OSError:
            return {"error": "git not found"}

        if check.returncode != 0:
            return {"error": "not a git repository"}

        # Run the actual command
        try:
            result = subprocess.run(
                ["git", *args],
                capture_output=True,
                timeout=_GIT_TIMEOUT,
                cwd=str(PROJECT_ROOT),
                text=True,
            )
        except subprocess.TimeoutExpired:
            logger.warning("git %r timed out after %ss", args, _GIT_TIMEOUT)
            return {"error": f"git command timed out after {_GIT_TIMEOUT}s"}
        except OSError as exc:
            return {"error": f"git execution failed: {exc}"}

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()

        return {
            "ok": True,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": result.returncode,
        }

    return await loop.run_in_executor(None, _do)
