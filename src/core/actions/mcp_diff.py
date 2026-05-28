"""mcp_diff tool — registered via @tool decorator.

Text diff comparison using Python's built-in difflib.

Actions:
  - **compare** — unified diff of two text strings (returns lines as list).
  - **ratio** — SequenceMatcher similarity score (0.0 – 1.0).
  - **files** — read two files from the data directory and diff them.
"""

from __future__ import annotations

import difflib
import logging
from pathlib import Path
from typing import Any

from src.config import settings
from src.core.actions.tool_registry import tool

logger = logging.getLogger(__name__)


@tool(
    name="mcp_diff",
    description=(
        "Compare text strings or files using difflib.\n\n"
        "Actions:\n"
        '- **compare** — unified diff of "text1" vs "text2" (returns lines).\n'
        "- **ratio** — SequenceMatcher similarity score (0.0 – 1.0).\n"
        "- **files** — read two files from data/ and diff them.\n\n"
        "Examples:\n"
        '  action="compare" text1="hello world" text2="hello there"\n'
        '  action="ratio" text1="hello" text2="hallo"\n'
        '  action="files" path1="a.txt" path2="b.txt"'
    ),
    category="utility",
    risk="low",
    params={
        "action": "str — 'compare', 'ratio' or 'files'",
        "text1": "str — first text (required for compare/ratio)",
        "text2": "str — second text (required for compare/ratio)",
        "path1": "str — relative path inside data/ to first file (required for files)",
        "path2": "str — relative path inside data/ to second file (required for files)",
    },
)
async def mcp_diff(
    action: str = "",
    text1: str = "",
    text2: str = "",
    path1: str = "",
    path2: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """Compare text strings or files using Python's built-in difflib."""
    try:
        if action == "compare":
            return _do_compare(text1, text2)
        elif action == "ratio":
            return _do_ratio(text1, text2)
        elif action == "files":
            return await _do_files(path1, path2)
        else:
            return {
                "error": (
                    f"Unknown action {action!r}. Valid actions: compare, ratio, files"
                )
            }
    except ValueError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        logger.exception("mcp_diff(%r) failed", action)
        return {"error": f"Unexpected error: {exc}"}


# ── Helpers ──────────────────────────────────────────────────────────────


def _do_compare(text1: str, text2: str) -> dict[str, Any]:
    if not text1 and not text2:
        return {"error": "At least one of text1/text2 must be non-empty"}
    lines1 = text1.splitlines(keepends=True)
    lines2 = text2.splitlines(keepends=True)
    diff = list(difflib.unified_diff(lines1, lines2, lineterm="\n"))
    return {
        "ok": True,
        "action": "compare",
        "lines": diff,
        "line_count": len(diff),
    }


def _do_ratio(text1: str, text2: str) -> dict[str, Any]:
    if not text1 and not text2:
        return {"error": "At least one of text1/text2 must be non-empty"}
    matcher = difflib.SequenceMatcher(None, text1, text2)
    ratio = matcher.ratio()
    return {
        "ok": True,
        "action": "ratio",
        "ratio": round(ratio, 6),
    }


async def _do_files(path1: str, path2: str) -> dict[str, Any]:
    if not path1 or not path2:
        return {"error": "Both path1 and path2 are required for action='files'"}

    data_dir: Path = settings.data_dir

    file1 = _resolve_path(data_dir, path1)
    file2 = _resolve_path(data_dir, path2)

    if not file1.exists():
        return {"error": f"File not found: {path1} (resolved: {file1})"}
    if not file2.exists():
        return {"error": f"File not found: {path2} (resolved: {file2})"}
    if not file1.is_file():
        return {"error": f"Not a file: {path1}"}
    if not file2.is_file():
        return {"error": f"Not a file: {path2}"}

    text1 = file1.read_text(encoding="utf-8", errors="replace")
    text2 = file2.read_text(encoding="utf-8", errors="replace")

    lines1 = text1.splitlines(keepends=True)
    lines2 = text2.splitlines(keepends=True)
    diff = list(difflib.unified_diff(lines1, lines2, lineterm="\n"))

    return {
        "ok": True,
        "action": "files",
        "file1": str(file1),
        "file2": str(file2),
        "lines": diff,
        "line_count": len(diff),
    }


def _resolve_path(data_dir: Path, user_path: str) -> Path:
    """Resolve a user-supplied relative path inside data_dir.

    Raises ValueError if the path attempts directory traversal.
    """
    resolved = (data_dir / user_path).resolve()
    # Ensure the resolved path stays within data_dir
    if not str(resolved).startswith(str(data_dir.resolve())):
        raise ValueError(f"Path {user_path!r} escapes the data directory")
    return resolved
