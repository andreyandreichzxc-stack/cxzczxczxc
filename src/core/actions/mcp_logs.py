"""mcp_logs tool — registered via @tool decorator.

Log file reading tool.

Actions:
- ``action="tail" path="data/bot.log" lines=50`` — read last N lines (memory-safe)
- ``action="grep" path="data/bot.log" pattern="ERROR" lines=20`` — grep for pattern
- ``action="size" path="data/bot.log"`` — file size in human-readable format

Path validation ensures only ``settings.data_dir`` is accessible.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any

from src.config import settings
from src.core.actions.tool_registry import tool

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

_TAIL_CHUNK_SIZE = 4096  # bytes per chunk when reading from end
_MAX_TAIL_LINES = 1000
_MAX_GREP_LINES = 500
_MAX_FILE_SIZE_MB = 100  # refuse files larger than this


# ══════════════════════════════════════════════════════════════════════════
# Tool: mcp_logs
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="mcp_logs",
    description=(
        "Read and search log files.  Supports three actions:\n"
        "- 'tail' — read last N lines from a log file (memory-safe).\n"
        "- 'grep' — search a log file for a regex pattern.\n"
        "- 'size' — get file size in human-readable format.\n"
        "Only files under the data/ directory are accessible."
    ),
    category="system",
    risk="low",
    params={
        "action": "str — 'tail', 'grep', or 'size'",
        "path": "str — path to a log file (required)",
        "lines": "int — number of lines (default 50, max 1000 for 'tail', max 500 for 'grep')",
        "pattern": "str — regex pattern to search for (required for 'grep')",
    },
)
async def mcp_logs(
    action: str,
    path: str = "",
    lines: int = 50,
    pattern: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """Log file reading tool.

    Args:
        action: ``"tail"``, ``"grep"``, or ``"size"``.
        path: Path to a log file (required).
        lines: Number of lines (default 50).
        pattern: Regex pattern (required for ``action="grep"``).

    Returns:
        A dict with file contents or an ``"error"`` key.
    """
    try:
        if action not in ("tail", "grep", "size"):
            return {
                "error": f"Unknown action {action!r}. Valid actions: tail, grep, size"
            }

        if not path or not path.strip():
            return {"error": "path parameter is required"}

        resolved = _safe_log_path(path.strip())
        if resolved is None:
            return {"error": f"Path {path!r} is outside allowed directories"}

        if action == "size":
            return await _file_size(resolved)
        elif action == "tail":
            n = max(1, min(lines, _MAX_TAIL_LINES))
            return await _tail_file(resolved, n)
        else:  # grep
            if not pattern:
                return {"error": "pattern parameter is required for action='grep'"}
            n = max(1, min(lines, _MAX_GREP_LINES))
            return await _grep_file(resolved, pattern, n)
    except Exception as exc:
        logger.exception("mcp_logs(%r) failed", action)
        return {"error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════


def _safe_log_path(raw: str) -> Path | None:
    """Resolve *raw* to an absolute path within ``settings.data_dir``."""
    normalised = raw.replace("/", os.sep).replace("\\", os.sep)
    if ".." in normalised.split(os.sep):
        return None

    resolved = Path(raw).resolve()
    data_dir = settings.data_dir.resolve()

    try:
        resolved.relative_to(data_dir)
    except ValueError:
        return None

    return resolved


def _human_size(size_bytes: int) -> str:
    """Format bytes into a human-readable string."""
    val = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if val < 1024:
            return f"{val:.1f} {unit}"
        val /= 1024
    return f"{val:.1f} TB"


# ══════════════════════════════════════════════════════════════════════════
# Action implementations
# ══════════════════════════════════════════════════════════════════════════


async def _file_size(file_path: Path) -> dict[str, Any]:
    """Get the file size."""

    def _do() -> dict[str, Any]:
        if not file_path.is_file():
            return {"error": "File not found"}
        size_bytes = file_path.stat().st_size
        return {
            "ok": True,
            "path": str(file_path),
            "size_bytes": size_bytes,
            "size_human": _human_size(size_bytes),
        }

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _do)


async def _tail_file(file_path: Path, lines: int) -> dict[str, Any]:
    """Memory-safe tail: seek from end of file, read chunks backwards."""

    def _do() -> dict[str, Any]:
        if not file_path.is_file():
            return {"error": "File not found"}

        file_size = file_path.stat().st_size
        if file_size > _MAX_FILE_SIZE_MB * 1024 * 1024:
            return {
                "error": f"File too large ({_human_size(file_size)}) — "
                f"refusing to process"
            }

        # Edge case: empty file
        if file_size == 0:
            return {"ok": True, "lines": [], "count": 0, "total_lines": 0}

        collected: list[str] = []
        chunks_read = 0
        pos = file_size

        with open(str(file_path), "rb") as f:
            while pos > 0 and len(collected) < lines:
                # How many bytes to read this chunk
                read_size = min(_TAIL_CHUNK_SIZE, pos)
                pos -= read_size
                f.seek(pos)
                chunk = f.read(read_size)
                chunks_read += 1

                # Decode chunk and split into lines
                try:
                    text = chunk.decode("utf-8", errors="replace")
                except Exception:
                    text = chunk.decode("latin-1", errors="replace")

                chunk_lines = text.splitlines(keepends=True)

                # Prepend to our collected buffer
                collected = chunk_lines + collected

                # If we didn't read a full chunk's worth of lines,
                # continue reading more chunks
                if len(collected) >= lines:
                    break

            # Trim to exactly `lines` items, but keep the first
            # (potentially partial) line whole — we don't want
            # to split in the middle of a line
            if len(collected) > lines:
                # The first line (index 0) may be partial — but we keep it
                # and drop lines from the beginning (which are already partial)
                collected = collected[-lines:]

        # Join and re-split to get clean lines
        full_text = "".join(collected)
        final_lines = full_text.splitlines()

        # Count total lines in file
        total_lines = 0
        try:
            with open(str(file_path), "rb") as f:
                for _ in f:
                    total_lines += 1
        except Exception:
            total_lines = len(final_lines)

        return {
            "ok": True,
            "lines": final_lines[-lines:],
            "count": min(len(final_lines), lines),
            "total_lines": total_lines,
            "path": str(file_path),
        }

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _do)


async def _grep_file(file_path: Path, pattern: str, lines: int) -> dict[str, Any]:
    """Grep for *pattern* in *file_path*, returning matching lines."""

    def _do() -> dict[str, Any]:
        if not file_path.is_file():
            return {"error": "File not found"}

        file_size = file_path.stat().st_size
        if file_size > _MAX_FILE_SIZE_MB * 1024 * 1024:
            return {
                "error": f"File too large ({_human_size(file_size)}) — "
                f"refusing to process"
            }

        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            return {"error": f"Invalid regex pattern: {exc}"}

        matches: list[dict[str, Any]] = []
        try:
            with open(str(file_path), "r", encoding="utf-8", errors="replace") as f:
                for line_no, line in enumerate(f, 1):
                    if regex.search(line):
                        matches.append(
                            {
                                "line": line_no,
                                "content": line.rstrip("\n\r"),
                            }
                        )
                        if len(matches) >= lines:
                            break
        except Exception as exc:
            return {"error": f"Failed to read file: {exc}"}

        return {
            "ok": True,
            "pattern": pattern,
            "matches": matches,
            "count": len(matches),
            "limited": len(matches) == lines,
            "path": str(file_path),
        }

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _do)
