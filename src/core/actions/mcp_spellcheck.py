"""mcp_spellcheck tool — registered via @tool decorator.

Spell checking via language-tool-python (LanguageTool).

Actions:
- ``action="check" text="превед медвед" lang="ru"`` — check spelling.
- ``action="languages"`` — list available language codes.

language-tool-python is imported lazily.  LanguageTool instances are cached
at module level because initialisation is slow.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.core.actions.tool_registry import tool

logger = logging.getLogger(__name__)

# ── Module-level cache ───────────────────────────────────────────────────
# LanguageTool initialisation is expensive (downloads grammar models).
# Cache one instance per language code at module level.

_LT_INSTANCES: dict[str, Any] = {}
_LT_SUPPORTED_LANGUAGES: set[str] | None = None


def _get_language_tool(lang: str) -> Any:
    """Get or create a cached LanguageTool instance for *lang*.

    Raises ImportError if language-tool-python is not installed.
    """
    if lang not in _LT_INSTANCES:
        import language_tool_python  # type: ignore[import-untyped]  # lazy import

        _LT_INSTANCES[lang] = language_tool_python.LanguageTool(lang)
    return _LT_INSTANCES[lang]


def _get_supported_languages() -> set[str]:
    """Return cached set of supported language codes.

    Falls back to a common set if the module is not installed or the
    ``list_languages`` API is unavailable.
    """
    global _LT_SUPPORTED_LANGUAGES

    if _LT_SUPPORTED_LANGUAGES is not None:
        return _LT_SUPPORTED_LANGUAGES

    # Try to discover from language_tool_python
    try:
        import language_tool_python  # type: ignore[import-untyped]

        try:
            langs = language_tool_python.LanguageTool.list_languages()
            _LT_SUPPORTED_LANGUAGES = set(langs)
            return _LT_SUPPORTED_LANGUAGES
        except AttributeError:
            pass
        except Exception:
            logger.debug("Could not list languages via API", exc_info=True)
    except ImportError:
        pass

    # Fallback — common languages supported by LanguageTool
    _LT_SUPPORTED_LANGUAGES = {
        "en",
        "ru",
        "de",
        "fr",
        "es",
        "it",
        "pt",
        "nl",
        "pl",
        "uk",
        "ca",
        "da",
        "el",
        "eo",
        "fa",
        "fi",
        "ga",
        "he",
        "hi",
        "hr",
        "hu",
        "id",
        "ja",
        "ko",
        "nb",
        "nn",
        "ro",
        "sk",
        "sl",
        "sr",
        "sv",
        "ta",
        "tl",
        "tr",
        "zh",
    }
    return _LT_SUPPORTED_LANGUAGES


# ══════════════════════════════════════════════════════════════════════════
# Tool: mcp_spellcheck
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="mcp_spellcheck",
    description=(
        "Check spelling and grammar via LanguageTool.\n\n"
        "Actions:\n"
        "- **check** — check text; returns list of issues with suggestions.\n"
        "- **languages** — list available language codes.\n\n"
        "Examples:\n"
        '  action="check" text="превед медвед" lang="ru"\n'
        '  action="check" text="Helo world" lang="en"\n'
        '  action="languages"'
    ),
    category="utility",
    risk="low",
    params={
        "action": "str — 'check' or 'languages'",
        "text": "str — text to check (required for 'check')",
        "lang": "str — language code (default 'ru'; also 'en'; used with 'check')",
    },
)
async def mcp_spellcheck(
    action: str = "",
    text: str = "",
    lang: str = "ru",
    **kwargs: Any,
) -> dict[str, Any]:
    """Spell checking tool via LanguageTool."""
    try:
        if action not in ("check", "languages"):
            return {
                "error": (f"Unknown action {action!r}. Valid actions: check, languages")
            }

        if action == "languages":
            return await _list_languages()
        else:  # check
            if not text or not text.strip():
                return {"error": "text parameter is required for action='check'"}
            return await _check_text(text, lang.strip().lower())
    except ImportError:
        return {
            "error": (
                "language-tool-python not installed: pip install language-tool-python. "
                "Note: language-tool-python requires a Java runtime "
                "(https://java.com) to be installed."
            )
        }
    except Exception as exc:
        logger.exception("mcp_spellcheck(%r) failed", action)
        return {"error": f"Unexpected error: {exc}"}


# ══════════════════════════════════════════════════════════════════════════
# Action implementations
# ══════════════════════════════════════════════════════════════════════════


async def _check_text(text: str, lang: str) -> dict[str, Any]:
    """Check spelling and grammar of *text* in *lang*."""
    # Validate language support
    supported = _get_supported_languages()
    if lang not in supported:
        return {
            "error": (
                f"Unsupported language {lang!r}. "
                f"Supported: {', '.join(sorted(supported))}"
            )
        }

    # Get or create LanguageTool instance (raises ImportError if not installed)
    try:
        tool_instance = _get_language_tool(lang)
    except ImportError:
        raise
    except Exception as exc:
        return {"error": f"Failed to initialise LanguageTool: {exc}"}

    loop = asyncio.get_running_loop()

    def _check() -> list[dict[str, Any]]:
        matches = tool_instance.check(text)
        results: list[dict[str, Any]] = []
        for m in matches:
            results.append(
                {
                    "word": text[m.offset : m.offset + m.errorLength],
                    "suggestions": m.replacements[:5],
                    "offset": m.offset,
                    "length": m.errorLength,
                    "message": m.message,
                    "rule_id": m.ruleId,
                    "rule_category": m.ruleCategory or "",
                }
            )
        return results

    try:
        matches = await loop.run_in_executor(None, _check)
    except Exception as exc:
        logger.warning("Spell check error: %s", exc)
        return {"error": f"Spell check failed: {exc}"}

    return {
        "ok": True,
        "action": "check",
        "lang": lang,
        "text_length": len(text),
        "issues": matches,
        "issue_count": len(matches),
    }


async def _list_languages() -> dict[str, Any]:
    """Return available LanguageTool language codes."""
    supported = _get_supported_languages()
    languages = sorted(supported)

    return {
        "ok": True,
        "action": "languages",
        "languages": languages,
        "count": len(languages),
    }
