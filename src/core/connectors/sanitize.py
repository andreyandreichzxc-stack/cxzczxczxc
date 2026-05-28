"""Sanitization helpers for untrusted connector content."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any


INVISIBLE_CHARS = re.compile(
    "["
    "\u200b"
    "\u200c"
    "\u200d"
    "\u200e"
    "\u200f"
    "\u2028"
    "\u2029"
    "\u202a-\u202e"
    "\u2060"
    "\u2061-\u2064"
    "\ufeff"
    "\ufff9-\ufffb"
    "]"
)
EXCESSIVE_NEWLINES = re.compile(r"\n{3,}")


def sanitize_text(value: str | None, *, max_length: int = 4096) -> str:
    if not value:
        return ""

    cleaned: list[str] = []
    for char in value:
        category = unicodedata.category(char)
        if category in {"Cc", "Cf"}:
            if char in {"\n", "\t"}:
                cleaned.append(char)
            continue
        cleaned.append(char)

    text = "".join(cleaned)
    text = INVISIBLE_CHARS.sub("", text)
    text = EXCESSIVE_NEWLINES.sub("\n\n", text).strip()
    if len(text) > max_length:
        return f"{text[:max_length]}... [truncated]"
    return text


def sanitize_untrusted(value: Any, *, max_string_length: int = 4096) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): sanitize_untrusted(item, max_string_length=max_string_length)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [sanitize_untrusted(item, max_string_length=max_string_length) for item in value]
    if isinstance(value, str):
        return sanitize_text(value, max_length=max_string_length)
    return value
