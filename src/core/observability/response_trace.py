"""Structured, secret-safe trace events for assistant responses."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)

_SECRET_KEYS = (
    "api_key",
    "authorization",
    "bot_token",
    "cookie",
    "password",
    "secret",
    "token",
)


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if any(secret in key_text for secret in _SECRET_KEYS):
                redacted[str(key)] = "***"
            else:
                redacted[str(key)] = _redact(item)
        return redacted
    if isinstance(value, (list, tuple, set)):
        return [_redact(item) for item in value]
    if isinstance(value, str) and len(value) > 500:
        return value[:500] + "..."
    return value


def _count_memory_facts(memory_context: str | None) -> int:
    if not memory_context:
        return 0
    count = 0
    for line in memory_context.splitlines():
        stripped = line.strip()
        if stripped.startswith(("-", "*", "•", "[")):
            count += 1
    return count


def _context_sources(memory_context: str | None) -> list[str]:
    if not memory_context:
        return []
    sources: set[str] = set()
    for marker in ("recall_context", "context_engine", "self_profile"):
        if marker in memory_context:
            sources.add(marker)
    for line in memory_context.splitlines():
        if line.startswith("[") and "]" in line:
            sources.add(line[1 : line.index("]")].split(":", 1)[0])
    return sorted(sources)


def _tool_names(items: Iterable[Any] | None) -> list[str]:
    names: list[str] = []
    for item in items or []:
        if isinstance(item, dict):
            value = item.get("tool") or item.get("intent") or item.get("action")
            if value:
                names.append(str(value))
        elif item:
            names.append(str(item))
    return names[:20]


def log_response_trace(
    *,
    route: str,
    owner_id: int | None = None,
    memory_context: str | None = None,
    context_sources: Iterable[str] | None = None,
    tools_proposed: Iterable[Any] | None = None,
    tools_executed: Iterable[Any] | None = None,
    tools_blocked: Iterable[Any] | None = None,
    guardrail_decision: dict[str, Any] | None = None,
    humanizer_mode: str = "off",
    humanizer_changed: bool = False,
    extra: dict[str, Any] | None = None,
) -> None:
    """Emit a compact response trace without user text or secrets."""

    sources = set(context_sources or [])
    sources.update(_context_sources(memory_context))
    payload = {
        "route": route,
        "owner_id": owner_id,
        "context_sources": sorted(sources),
        "memory_facts_count": _count_memory_facts(memory_context),
        "tools_proposed": _tool_names(tools_proposed),
        "tools_executed": _tool_names(tools_executed),
        "tools_blocked": _tool_names(tools_blocked),
        "guardrail_decision": _redact(guardrail_decision or {}),
        "humanizer": {
            "mode": humanizer_mode,
            "changed": humanizer_changed,
        },
        "extra": _redact(extra or {}),
    }
    logger.info("response_trace", extra={"response_trace": payload})
