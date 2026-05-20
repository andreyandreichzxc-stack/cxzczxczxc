"""LRU-кэш эмбеддингов чтобы не перевычислять для одинаковых текстов."""

from collections import OrderedDict
import hashlib

_cache: OrderedDict = OrderedDict()
MAX_SIZE = 500


def _hash(text: str, model: str = "") -> str:
    raw = f"{model}||{text}" if model else text
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get(text: str, model: str = "") -> list[float] | None:
    key = _hash(text, model)
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    return None


def set(text: str, embedding: list[float], model: str = "") -> None:
    key = _hash(text, model)
    if key in _cache:
        _cache.move_to_end(key)
        _cache[key] = embedding
    else:
        if len(_cache) >= MAX_SIZE:
            _cache.popitem(last=False)
        _cache[key] = embedding


def clear() -> None:
    _cache.clear()
