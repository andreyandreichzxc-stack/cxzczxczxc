"""mcp_notify tool — registered via @tool decorator.

Sends OS-level desktop notifications and beeps.

Actions:
- **send** — pop a desktop notification (via ``plyer`` if available, else log).
- **beep** — emit ``count`` audible beeps (via ``print('\\a')``).
"""

from __future__ import annotations

import logging
from typing import Any

from src.core.actions.tool_registry import tool

logger = logging.getLogger(__name__)

# ── Platform-specific beep support ──────────────────────────────────────────

try:
    import winsound  # type: ignore[import-untyped]
except ImportError:
    winsound = None  # Linux/macOS — no winsound available


# ══════════════════════════════════════════════════════════════════════════
# Tool: mcp_notify
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="mcp_notify",
    description=(
        "Send OS-level desktop notifications or audible beeps. "
        "Supports two actions:\n"
        "- 'send' — pop a desktop notification with title and message.\n"
        "- 'beep' — emit an audible beep (repeated *count* times)."
    ),
    category="utility",
    risk="low",
    params={
        "action": "str — 'send' or 'beep'",
        "title": "str — notification title (required for 'send')",
        "message": "str — notification body text (required for 'send')",
        "count": "int — number of beeps (default 3, used with 'beep')",
    },
)
async def mcp_notify(
    action: str,
    title: str = "",
    message: str = "",
    count: int = 3,
    **kwargs: Any,
) -> dict[str, Any]:
    """OS notification tool.

    Args:
        action: ``"send"`` or ``"beep"``.
        title: Notification title (required for ``action="send"``).
        message: Notification body text (required for ``action="send"``).
        count: Number of beeps (default 3, used with ``action="beep"``).

    Returns:
        A dict with the result data or an ``"error"`` key on failure.
    """
    try:
        if action == "send":
            return await _notify_send(title, message)
        elif action == "beep":
            return _notify_beep(count)
        else:
            return {"error": f"Unknown action {action!r}. Valid actions: send, beep"}
    except Exception as exc:
        logger.exception("mcp_notify(%r) failed", action)
        return {"error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════
# Action implementations
# ══════════════════════════════════════════════════════════════════════════


async def _notify_send(title: str, message: str) -> dict[str, Any]:
    """Send a desktop notification, falling back to logging."""
    if not title or not title.strip():
        return {"error": "title parameter is required for action='send'"}
    if not message or not message.strip():
        return {"error": "message parameter is required for action='send'"}

    # Try plyer first
    try:
        from plyer import notification  # type: ignore[import-untyped]

        notification.notify(
            title=title.strip(),
            message=message.strip(),
            timeout=5,
        )
        logger.info("Desktop notification sent: %r — %r", title, message)
        return {"ok": True, "method": "plyer"}
    except ImportError:
        logger.info("plyer not available — falling back to log-only notification")
    except Exception as exc:
        logger.warning("plyer notification failed: %s — falling back to log", exc)

    # Fallback: just log it
    logger.info("NOTIFICATION: %r — %r", title, message)
    return {"ok": True, "method": "logging_only"}


def _notify_beep(count: int) -> dict[str, Any]:
    """Emit an audible beep via winsound (Windows) or terminal bell character."""
    count = max(1, min(count, 20))  # Sanity clamp

    if winsound:
        try:
            for _ in range(count):
                winsound.Beep(1000, 300)  # type: ignore[union-attr]
            return {"ok": True, "method": "winsound", "count": count}
        except Exception:
            pass

    # Fallback: print bell character
    beeps = "\a" * count
    print(beeps, end="", flush=True)  # noqa: T201

    return {
        "ok": True,
        "method": "bell_char",
        "count": count,
    }
