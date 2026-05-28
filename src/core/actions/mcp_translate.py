"""mcp_translate tool — registered via @tool decorator.

Translates text from one language to another using an available LLM provider.

Actions:
- **translate** — translate text from source language to target language via LLM.
"""

from __future__ import annotations

import logging
from typing import Any

from src.core.actions.tool_registry import tool
from src.llm.base import ChatMessage

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Tool: mcp_translate
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="mcp_translate",
    description=(
        "Translate text from one language to another using an LLM. "
        "Supports one action:\n"
        "- 'translate' — translate the given text to the target language.\n"
        "Example: action='translate' text='Hello' target_lang='ru'"
    ),
    category="utility",
    risk="low",
    params={
        "action": "str — 'translate'",
        "text": "str — text to translate",
        "target_lang": "str — target language code (e.g. 'ru', 'fr', 'de')",
        "source_lang": "str — source language code (default 'auto')",
    },
)
async def mcp_translate(
    action: str,
    text: str = "",
    target_lang: str = "",
    source_lang: str = "auto",
    **kwargs: Any,
) -> dict[str, Any]:
    """Translate text via LLM.

    Args:
        action: ``"translate"``.
        text: The text to translate.
        target_lang: Target language code (e.g. ``"ru"``, ``"fr"``, ``"de"``).
        source_lang: Source language code (default ``"auto"``).

    Keyword Args:
        provider: LLM provider with a ``chat()`` method (injected at runtime).

    Returns:
        A dict with ``"translated"`` text and language metadata or ``"error"``.
    """
    try:
        if action == "translate":
            return await _do_translate(text, target_lang, source_lang, kwargs)
        else:
            return {"error": f"Unknown action {action!r}. Valid actions: translate"}
    except Exception as exc:
        logger.exception("mcp_translate(%r) failed", action)
        return {"error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════
# Action implementation
# ══════════════════════════════════════════════════════════════════════════


async def _do_translate(
    text: str,
    target_lang: str,
    source_lang: str,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Perform translation via the injected LLM provider."""
    if not text or not text.strip():
        return {"error": "text parameter is required for action='translate'"}

    if not target_lang or not target_lang.strip():
        return {"error": "target_lang parameter is required"}

    provider = kwargs.get("provider")
    if provider is None:
        return {"error": "no LLM provider available"}

    source = source_lang.strip() or "auto"
    target = target_lang.strip()

    prompt = f"Translate from {source} to {target}: {text}"

    try:
        result = await provider.chat([ChatMessage(role="user", content=prompt)])
        translated = result.strip()
        if not translated:
            return {"error": "LLM returned empty translation"}
    except Exception as exc:
        logger.exception("Translation LLM call failed")
        return {"error": f"Translation failed: {exc}"}

    return {
        "ok": True,
        "translated": translated,
        "source_lang": source,
        "target_lang": target,
    }
