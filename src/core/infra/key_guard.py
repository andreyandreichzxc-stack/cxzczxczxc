"""Mask API keys in error strings and log messages."""

import re
import logging

logger = logging.getLogger(__name__)

# API key patterns
_KEY_PATTERNS = [
    r"sk-[A-Za-z0-9]{20,}",  # OpenAI/DeepSeek/etc
    r"sk-proj-[A-Za-z0-9]{20,}",  # OpenAI project key
    r"sk-ant-api03-[A-Za-z0-9_-]{20,}",  # Anthropic
    r"sk-or-[A-Za-z0-9]{20,}",  # OpenRouter
    r"AIza[A-Za-z0-9_-]{30,}",  # Gemini
    r"xai-[A-Za-z0-9]{20,}",  # Grok
    r"gsk_[A-Za-z0-9]{20,}",  # Groq
    r"dl-[A-Za-z0-9_-]{20,}",  # Deepgram
    r"Nb[A-Za-z0-9_-]{20,}",  # Mistral (some formats)
]

_MASKED_REPLACEMENT = "***"


def mask_keys(text: str) -> str:
    """Replace all API keys in string with ***."""
    if not text or not isinstance(text, str):
        return text
    for pattern in _KEY_PATTERNS:
        text = re.sub(pattern, _MASKED_REPLACEMENT, text)
    return text


def safe_str(exc: Exception) -> str:
    """Safe str() — masks keys."""
    return mask_keys(str(exc))


def safe_repr(exc: Exception) -> str:
    """Safe repr() — masks keys."""
    return mask_keys(repr(exc))
