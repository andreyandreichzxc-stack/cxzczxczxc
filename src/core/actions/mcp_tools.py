"""MCP Tools — lightweight filesystem and system tools for LLM interaction.

Provides two categories of tools registered via the ``@tool`` decorator:

- ``mcp_filesystem``: list, space, read, search operations on local files
- ``mcp_system``: status, version info about the bot runtime

Safety:
    All filesystem paths are validated against ``data_dir`` and ``PROJECT_ROOT``.
    Directory traversal (``..``) is explicitly blocked.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT, settings
from src.core.actions.tool_registry import tool

logger = logging.getLogger(__name__)


# ── Safe path resolution ───────────────────────────────────────────────

_ALLOWED_ROOTS: frozenset[Path] = frozenset(
    [
        settings.data_dir.resolve(),
        PROJECT_ROOT.resolve(),
    ]
)


def _safe_resolve(raw: str) -> Path | None:
    """Resolve *raw* to an absolute path, returning ``None`` if unsafe.

    Safety rules:
    1. The original *raw* must not contain ``..`` as a path component
       (directory traversal is explicitly forbidden).
    2. The resolved path must be under one of ``_ALLOWED_ROOTS``.
    """
    # Normalise separators so checks work on both Unix and Windows
    normalised = raw.replace("/", os.sep).replace("\\", os.sep)
    if ".." in normalised.split(os.sep):
        return None

    resolved = (PROJECT_ROOT / raw).resolve()

    for root in _ALLOWED_ROOTS:
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue

    return None


# ── Helper: detect text file ──────────────────────────────────────────

_TEXT_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",
        ".md",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".cfg",
        ".ini",
        ".conf",
        ".log",
        ".rst",
        ".html",
        ".css",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".sh",
        ".bat",
        ".ps1",
        ".env",
        ".gitignore",
        ".dockerignore",
        ".cfg",
        ".cfg",
    }
)


def _is_text_file(path: Path) -> bool:
    """Check if a file is likely text based on extension and content."""
    if path.suffix.lower() in _TEXT_EXTENSIONS:
        return True
    # Fallback: try to read a small chunk as UTF-8
    try:
        with open(path, "r", encoding="utf-8", errors="strict") as f:
            f.read(1024)
        return True
    except (UnicodeDecodeError, OSError):
        return False


# ══════════════════════════════════════════════════════════════════════════
# Tool 1: mcp_filesystem
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="mcp_filesystem",
    description=(
        "Access the bot's local filesystem. Supports actions:\n"
        "- 'list' — directory listing with names, sizes, and types\n"
        "- 'space' — disk usage statistics (free/total/used in GB)\n"
        "- 'read' — read first 2000 characters of a text file\n"
        "- 'search' — recursive grep for a pattern in text files\n"
        "Path is restricted to data/ and project root directory."
    ),
    category="system",
    risk="medium",
    params={
        "action": "str",
        "path": "str",
        "pattern": "str|None",
    },
)
async def mcp_filesystem(action: str, path: str = ".", **kwargs: Any) -> dict:
    """Filesystem introspection tool.

    Args:
        action: One of ``"list"``, ``"space"``, ``"read"``, ``"search"``.
        path: Directory or file path (relative to project root).
        pattern: Search regex pattern (only for ``action="search"``).

    Returns:
        A dict with the result data or an ``"error"`` key on failure.
    """
    try:
        if action == "list":
            return await _fs_list(path)
        elif action == "space":
            return await _fs_space()
        elif action == "read":
            return await _fs_read(path)
        elif action == "search":
            pattern = kwargs.get("pattern", "")
            if not pattern:
                return {"error": "Missing required param 'pattern' for action='search'"}
            return await _fs_search(path, pattern)
        else:
            return {
                "error": f"Unknown action {action!r}. Valid: list, space, read, search"
            }
    except Exception as exc:
        logger.exception("mcp_filesystem(%r, path=%r) failed", action, path)
        return {"error": str(exc)}


async def _fs_list(dir_path: str) -> dict:
    resolved = _safe_resolve(dir_path)
    if resolved is None:
        return {
            "error": f"Path {dir_path!r} is outside allowed directories or contains '..'"
        }
    if not resolved.is_dir():
        return {"error": f"Path {dir_path!r} is not a directory"}

    loop = asyncio.get_running_loop()

    def _scan() -> list[dict]:
        entries: list[dict] = []
        for entry in os.scandir(str(resolved)):
            st = entry.stat(follow_symlinks=False)
            entries.append(
                {
                    "name": entry.name,
                    "type": "dir" if entry.is_dir(follow_symlinks=False) else "file",
                    "size": st.st_size,
                }
            )
        entries.sort(key=lambda e: (e["type"] != "dir", e["name"].lower()))
        return entries

    entries = await loop.run_in_executor(None, _scan)
    return {
        "ok": True,
        "path": str(resolved),
        "entries": entries,
        "count": len(entries),
    }


async def _fs_space() -> dict:
    import shutil

    loop = asyncio.get_running_loop()
    usage = await loop.run_in_executor(
        None, lambda: shutil.disk_usage(settings.data_dir)
    )
    return {
        "ok": True,
        "total_gb": round(usage.total / (1024**3), 2),
        "used_gb": round(usage.used / (1024**3), 2),
        "free_gb": round(usage.free / (1024**3), 2),
        "used_pct": round(usage.used / usage.total * 100, 1),
    }


async def _fs_read(file_path: str) -> dict:
    resolved = _safe_resolve(file_path)
    if resolved is None:
        return {
            "error": f"Path {file_path!r} is outside allowed directories or contains '..'"
        }
    if not resolved.is_file():
        return {"error": f"Path {file_path!r} is not a file"}
    if not _is_text_file(resolved):
        return {"error": f"File {file_path!r} is not a text file (binary skipped)"}

    loop = asyncio.get_running_loop()

    def _read() -> tuple[str, int]:
        text = resolved.read_text(encoding="utf-8", errors="replace")
        return text[:2000], len(text)

    content, total_len = await loop.run_in_executor(None, _read)
    return {
        "ok": True,
        "path": str(resolved),
        "content": content,
        "truncated": total_len > 2000,
        "total_chars": total_len,
    }


async def _fs_search(dir_path: str, pattern: str) -> dict:
    resolved = _safe_resolve(dir_path)
    if resolved is None:
        return {
            "error": f"Path {dir_path!r} is outside allowed directories or contains '..'"
        }
    if not resolved.is_dir():
        return {"error": f"Path {dir_path!r} is not a directory"}

    limit = 50
    loop = asyncio.get_running_loop()

    def _search() -> list[dict]:
        matches: list[dict] = []
        compiled = re.compile(pattern, re.IGNORECASE)
        for root_str, _dirs, files in os.walk(str(resolved)):
            if len(matches) >= limit:
                break
            root = Path(root_str)
            for fname in files:
                if len(matches) >= limit:
                    break
                fp = root / fname
                if not _is_text_file(fp):
                    continue
                try:
                    text = fp.read_text(encoding="utf-8", errors="replace")
                    for lineno, line in enumerate(text.splitlines(), 1):
                        if compiled.search(line):
                            matches.append(
                                {
                                    "file": str(fp.relative_to(PROJECT_ROOT)),
                                    "line": lineno,
                                    "text": line.strip()[:200],
                                }
                            )
                            if len(matches) >= limit:
                                break
                except Exception:
                    continue
        return matches

    matches = await loop.run_in_executor(None, _search)
    return {
        "ok": True,
        "path": str(resolved),
        "pattern": pattern,
        "matches": matches,
        "count": len(matches),
    }


# ══════════════════════════════════════════════════════════════════════════
# Tool 2: mcp_system
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="mcp_system",
    description=(
        "Retrieve system-level information about the bot runtime. "
        "Supports actions:\n"
        "- 'status' — uptime, memory usage, platform, DB size\n"
        "- 'version' — SOUL.md title, git commit hash"
    ),
    category="system",
    risk="low",
    params={"action": "str"},
)
async def mcp_system(action: str, **kwargs: Any) -> dict:
    """System introspection tool.

    Args:
        action: One of ``"status"``, ``"version"``.

    Returns:
        A dict with the result data or an ``"error"`` key on failure.
    """
    try:
        if action == "status":
            return await _sys_status()
        elif action == "version":
            return await _sys_version()
        else:
            return {"error": f"Unknown action {action!r}. Valid: status, version"}
    except Exception as exc:
        logger.exception("mcp_system(%r) failed", action)
        return {"error": str(exc)}


async def _sys_status() -> dict:
    import platform as _platform
    import time as _time

    loop = asyncio.get_running_loop()
    info: dict[str, Any] = {}

    # --- Uptime & memory via psutil (optional dependency) ---
    def _collect_psutil() -> dict:
        import psutil  # type: ignore[import-untyped]

        boot_ts = psutil.boot_time()
        uptime = _time.time() - boot_ts
        mem = psutil.virtual_memory()
        return {
            "uptime_sec": round(uptime),
            "uptime_human": _format_uptime(uptime),
            "memory_total_mb": round(mem.total / (1024**2), 1),
            "memory_used_mb": round(mem.used / (1024**2), 1),
            "memory_available_mb": round(mem.available / (1024**2), 1),
            "memory_used_pct": mem.percent,
        }

    try:
        info.update(await loop.run_in_executor(None, _collect_psutil))
    except ImportError:
        logger.info("psutil not available — skipping memory/uptime collection")
        info["uptime_sec"] = None
        info["uptime_human"] = "N/A (install psutil)"
        info["memory_note"] = "Install psutil for memory/uptime info"
    except Exception as exc:
        logger.warning("psutil error: %s", exc)
        info["uptime_sec"] = None
        info["uptime_human"] = "N/A"
        info["memory_note"] = f"psutil error: {exc}"

    # --- Database file size ---
    db_path = settings.data_dir / "app.db"
    if db_path.is_file():
        info["db_size_mb"] = round(db_path.stat().st_size / (1024**2), 2)
    else:
        info["db_size_mb"] = None

    # --- Platform / Python ---
    info["platform"] = _platform.platform()
    info["python_version"] = _platform.python_version()

    return {"ok": True, **info}


async def _sys_version() -> dict:
    loop = asyncio.get_running_loop()
    result: dict[str, Any] = {}

    # --- SOUL.md title (first line) ---
    soul_path = PROJECT_ROOT / "SOUL.md"
    if soul_path.is_file():
        first_line = await loop.run_in_executor(
            None,
            lambda: (
                soul_path.read_text(encoding="utf-8", errors="replace")
                .split("\n")[0]
                .strip()
            ),
        )
        result["soul_title"] = first_line
    else:
        result["soul_title"] = None

    # --- Git commit hash ---
    git_dir = PROJECT_ROOT / ".git"
    if git_dir.is_dir():
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                "rev-parse",
                "--short",
                "HEAD",
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode == 0:
                result["git_commit"] = stdout.decode("utf-8").strip()
            else:
                result["git_commit"] = None
                result["git_error"] = stderr.decode("utf-8").strip()
        except FileNotFoundError:
            result["git_commit"] = None
            result["git_note"] = "git executable not found in PATH"
        except Exception as exc:
            result["git_commit"] = None
            result["git_error"] = str(exc)
    else:
        result["git_commit"] = None
        result["git_note"] = "Not a git repository"

    return {"ok": True, **result}


# ── Helpers ────────────────────────────────────────────────────────────


def _format_uptime(seconds: float) -> str:
    """Format uptime in seconds to a human-readable compact string."""
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
