"""Secret redaction helpers for connector inputs and outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


SECRET_KEY_PARTS = (
    "authorization",
    "cookie",
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
)

SECRET_KEY_NAMES = {
    "session",
    "session_id",
    "session_key",
    "session_secret",
    "session_token",
}

URL_SECRET_QUERY_KEY_PARTS = (
    "access_token",
    "auth",
    "bearer",
    "credential",
    "key",
    "secret",
    "signature",
    "sig",
    "token",
)


def is_secret_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    if lowered in SECRET_KEY_NAMES:
        return True
    return any(part in lowered for part in SECRET_KEY_PARTS)


def redact_value(value: Any) -> str:
    text = str(value)
    if len(text) <= 8:
        return "***"
    return f"{text[:3]}...{text[-3:]}"


def redact_url_secrets(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return value

    changed = False
    if parsed.username or parsed.password:
        host = parsed.hostname or ""
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        parsed = parsed._replace(netloc=f"***@{host}")
        changed = True

    redacted_query: list[tuple[str, str]] = []
    for key, item in parse_qsl(parsed.query, keep_blank_values=True):
        lowered = key.lower().replace("-", "_")
        if is_secret_key(key) or any(part in lowered for part in URL_SECRET_QUERY_KEY_PARTS):
            redacted_query.append((key, redact_value(item)))
            changed = True
        else:
            redacted_query.append((key, item))

    if not changed:
        return value
    return urlunparse(parsed._replace(query=urlencode(redacted_query, doseq=True)))


def redact_secrets(value: Any) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            redacted[key_text] = redact_value(item) if is_secret_key(key_text) else redact_secrets(item)
        return redacted
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact_secrets(item) for item in value]
    if isinstance(value, str):
        return redact_url_secrets(value)
    return value
