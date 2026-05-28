"""mcp_env tool — registered via @tool decorator.

Environment variable inspection (read-only except session-local set).

Actions:
- ``action="get" key="BOT_TOKEN"`` — get a single env var (sensitive keys masked)
- ``action="list" prefix=""`` — list all env vars (all values masked)
- ``action="set" key="MY_VAR" value="hello"`` — set a session-local env var

Sensitive keys (BOT_TOKEN, API_KEY, SECRET, PASSWORD, TOKEN, ENCRYPTION_KEY,
CREDENTIALS) are always masked.  Set is session-only (not persisted).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.core.actions.tool_registry import ToolActionSpec, tool

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

_SENSITIVE_KEY_PATTERNS = frozenset(
    {
        "BOT_TOKEN",
        "API_KEY",
        "SECRET",
        "PASSWORD",
        "TOKEN",
        "ENCRYPTION_KEY",
        "CREDENTIALS",
    }
)


# ══════════════════════════════════════════════════════════════════════════
# Tool: mcp_env
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="mcp_env",
    description=(
        "Inspect and modify environment variables.  Supports three actions:\n"
        "- 'get' — get the value of an environment variable "
        "(sensitive keys are masked).\n"
        "- 'list' — list all environment variables (all values masked).\n"
        "- 'set' — set a session-local env var (not persisted).\n"
        "Sensitive keys: BOT_TOKEN, API_KEY, SECRET, PASSWORD, TOKEN, "
        "ENCRYPTION_KEY, CREDENTIALS — always masked."
    ),
    category="system",
    risk="medium",
    requires_confirmation=True,
    actions={
        "get": ToolActionSpec(
            name="get",
            risk="low",
            read_only=True,
            idempotent=True,
            user_content=False,
        ),
        "list": ToolActionSpec(
            name="list",
            risk="low",
            read_only=True,
            idempotent=True,
            user_content=False,
        ),
        "set": ToolActionSpec(
            name="set",
            risk="high",
            read_only=False,
            destructive=False,
            idempotent=False,
            requires_confirmation=True,
            user_content=False,
        ),
    },
    params={
        "action": "str — 'get', 'list', or 'set'",
        "key": "str — environment variable name (required for 'get' and 'set')",
        "value": "str — value to set (required for 'set')",
        "prefix": "str — filter prefix (optional, used with 'list')",
    },
)
async def mcp_env(
    action: str,
    key: str = "",
    value: str = "",
    prefix: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """Environment variable tool.

    Args:
        action: ``"get"``, ``"list"``, or ``"set"``.
        key: Environment variable name.
        value: Value to set (for ``action="set"``).
        prefix: Filter prefix (for ``action="list"``).

    Returns:
        A dict with env info or an ``"error"`` key.
    """
    try:
        if action not in ("get", "list", "set"):
            return {
                "error": f"Unknown action {action!r}. Valid actions: get, list, set"
            }

        if action == "get":
            if not key or not key.strip():
                return {"error": "key parameter is required for action='get'"}
            return _get_env(key.strip())
        elif action == "list":
            return _list_env(prefix.strip())
        else:  # set
            if not key or not key.strip():
                return {"error": "key parameter is required for action='set'"}
            if not value:
                return {"error": "value parameter is required for action='set'"}
            if not bool(kwargs.get("_confirmed", False)):
                return {"error": "requires confirmation"}
            return _set_env(key.strip(), value)
    except Exception as exc:
        logger.exception("mcp_env(%r) failed", action)
        return {"error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════


def _is_sensitive(key: str) -> bool:
    """Check whether *key* matches any sensitive-key pattern."""
    upper = key.upper()
    for pattern in _SENSITIVE_KEY_PATTERNS:
        if pattern in upper:
            return True
    return False


def _mask(value: str) -> str:
    """Mask a value, showing only first 4 + last 4 characters."""
    if len(value) <= 8:
        return value[:2] + "****" + value[-2:] if len(value) > 4 else "****"
    return value[:4] + "****" + value[-4:]


# ══════════════════════════════════════════════════════════════════════════
# Action implementations
# ══════════════════════════════════════════════════════════════════════════


def _get_env(key: str) -> dict[str, Any]:
    """Get a single environment variable."""
    value = os.environ.get(key)
    if value is None:
        return {"ok": True, "key": key, "found": False}

    display = _mask(value)

    return {
        "ok": True,
        "key": key,
        "found": True,
        "value": display,
        "masked": True,
    }


def _list_env(prefix: str) -> dict[str, Any]:
    """List environment variables, optionally filtered by *prefix*."""
    items: list[dict[str, Any]] = []
    for env_key, env_value in sorted(os.environ.items()):
        if prefix and not env_key.upper().startswith(prefix.upper()):
            continue

        # Always mask — even non-sensitive keys
        display = _mask(env_value)

        items.append(
            {
                "key": env_key,
                "value": display,
                "masked": True,  # Always mask in list view
            }
        )
    return {
        "ok": True,
        "count": len(items),
        "prefix": prefix or "(all)",
        "variables": items,
    }


def _set_env(key: str, value: str) -> dict[str, Any]:
    """Set a session-local environment variable.

    This is NOT persisted — it only affects the current process.
    """
    os.environ[key] = value
    logger.info("Environment variable %s set (session-local)", key)
    return {
        "ok": True,
        "key": key,
        "message": f"Environment variable {key!r} set (session-local, not persisted)",
    }
