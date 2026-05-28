"""mcp_json tool — registered via @tool decorator.

JSON formatting, validation, and simple dot-notation path extraction.

Features:
- ``action="format"`` — pretty-print JSON with ``indent=2``.
- ``action="validate"`` — check whether a string is valid JSON.
- ``action="path"`` — extract a value at a dot-notation path
  (e.g. ``"a.b"`` from ``{"a": {"b": 1}}``).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from src.core.actions.tool_registry import ToolActionSpec, tool

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Tool: mcp_json
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="mcp_json",
    description=(
        "JSON formatting, validation, and dot-notation path extraction.  "
        "Supports three actions:\n"
        "- 'format' — pretty-print JSON with indentation.\n"
        "- 'validate' — check if a string is valid JSON.\n"
        "- 'path' — extract a value at a simple dot-notation path "
        "(e.g. 'a.b' from '{\"a\": {\"b\": 1}}')."
    ),
    category="utility",
    risk="low",
    actions={
        "format": ToolActionSpec(
            name="format",
            risk="low",
            read_only=True,
            idempotent=True,
            user_content=True,
        ),
        "validate": ToolActionSpec(
            name="validate",
            risk="low",
            read_only=True,
            idempotent=True,
            user_content=True,
        ),
        "path": ToolActionSpec(
            name="path",
            risk="low",
            read_only=True,
            idempotent=True,
            user_content=True,
        ),
    },
    params={
        "action": "str — 'format', 'validate', or 'path'",
        "data": "str — JSON string to operate on",
        "path": "str — dot-notation path (required for action='path')",
    },
)
async def mcp_json(
    action: str,
    data: str = "",
    path: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """JSON utility tool.

    Args:
        action: ``"format"``, ``"validate"``, or ``"path"``.
        data: JSON string to operate on.
        path: Dot-notation path (required for ``action="path"``).

    Returns:
        A dict with the result or an ``"error"`` key on failure.
    """
    try:
        if action not in ("format", "validate", "path"):
            return {
                "error": f"Unknown action {action!r}. "
                f"Valid actions: format, validate, path"
            }

        if not data:
            return {"error": "data parameter is required"}

        if action == "format":
            return await _format_json(data)
        elif action == "validate":
            return await _validate_json(data)
        else:
            if not path:
                return {"error": "path parameter is required for action='path'"}
            return await _path_json(data, path.strip())
    except Exception as exc:
        logger.exception("mcp_json(%r) failed", action)
        return {"error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════
# Action implementations
# ══════════════════════════════════════════════════════════════════════════


async def _format_json(data: str) -> dict[str, Any]:
    """Pretty-print *data* as indented JSON."""

    def _do() -> dict[str, Any]:
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as exc:
            return {"error": f"Invalid JSON: {exc}"}

        formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
        return {
            "ok": True,
            "formatted": formatted,
            "chars": len(formatted),
        }

    return await _run_sync(_do)


async def _validate_json(data: str) -> dict[str, Any]:
    """Check whether *data* is valid JSON."""

    def _do() -> dict[str, Any]:
        try:
            json.loads(data)
            return {"ok": True, "valid": True}
        except json.JSONDecodeError as exc:
            return {
                "ok": True,
                "valid": False,
                "error": str(exc),
            }

    return await _run_sync(_do)


async def _path_json(data: str, path: str) -> dict[str, Any]:
    """Extract a value at *path* (dot-notation) from *data*."""

    def _do() -> dict[str, Any]:
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as exc:
            return {"error": f"Invalid JSON: {exc}"}

        keys = path.split(".")
        current: Any = parsed

        for key in keys:
            if not key:
                return {"error": f"Invalid path segment empty in {path!r}"}
            if isinstance(current, dict) and key in current:
                current = current[key]
            elif isinstance(current, list) and key.isdigit():
                idx = int(key)
                if 0 <= idx < len(current):
                    current = current[idx]
                else:
                    return {
                        "error": f"List index {idx} out of range (length {len(current)})"
                    }
            else:
                return {"error": f"Key {key!r} not found at path {path!r}"}

        result_json = json.dumps(current, indent=2, ensure_ascii=False)
        return {
            "ok": True,
            "path": path,
            "value": current,
            "formatted": result_json,
        }

    return await _run_sync(_do)


# ── Sync runner ──────────────────────────────────────────────────────────


async def _run_sync(fn: Any) -> dict[str, Any]:
    """Run a synchronous function in a thread-pool executor.

    All the JSON operations are CPU-fast so this is mostly a formality
    to keep the async contract — no ``asyncio`` is actually needed here.
    """
    loop = None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        pass

    if loop is not None:
        return await loop.run_in_executor(None, fn)
    return fn()  # fallback for synchronous contexts
