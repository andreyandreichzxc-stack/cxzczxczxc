"""In-memory TTL-кэш с инвалидацией по префиксу.

Синглтон на уровне модуля. Все операции синхронные.
TTL отсчитывается через time.monotonic().
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_cache: dict[str, tuple[float, Any]] = {}
"""Кэш: ключ -> (expires_at_monotonic, value)"""


def get(key: str) -> Any | None:
    """Вернуть значение из кэша или None если ключа нет / TTL истёк."""
    entry = _cache.get(key)
    if entry is None:
        return None

    expires_at, value = entry
    if time.monotonic() < expires_at:
        return value

    # TTL истёк — удалить
    del _cache[key]
    logger.debug("Cache expired: %s", key)
    return None


def put(key: str, value: Any, ttl: int = 30) -> None:
    """Сохранить значение в кэш с TTL в секундах (по умолчанию 30)."""
    expires_at = time.monotonic() + ttl
    _cache[key] = (expires_at, value)
    logger.debug("Cache put: %s (ttl=%ds)", key, ttl)


def invalidate(prefix: str = "") -> None:
    """Очистить кэш.

    Если prefix пуст — очистить весь кэш.
    Иначе — удалить только ключи, начинающиеся с prefix.
    """
    if not prefix:
        _cache.clear()
        logger.debug("Cache fully invalidated")
        return

    keys_to_del = [k for k in _cache if k.startswith(prefix)]
    for k in keys_to_del:
        del _cache[k]

    if keys_to_del:
        logger.debug("Cache invalidated: prefix=%s (%d keys)", prefix, len(keys_to_del))
