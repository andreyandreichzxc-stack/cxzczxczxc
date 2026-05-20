"""Детектор фраз отсутствия в исходящих сообщениях."""

import re
from typing import Optional

_AWAY_PATTERNS = [
    r"я уш[ёе]л",
    r"я не дома",
    r"меня не будет",
    r"буду позже",
    r"отхожу",
    r"ухожу",
    r"пока",
    r"до вечера",
    r"я спать",
    r"спокойной ночи",
    r"отключаюсь",
]

_SOON_BACK_PATTERNS = [
    r"скоро буду",
    r"через .* буду",
    r"подъезжаю",
    r"еду",
    r"почти на месте",
    r"возвращаюсь",
]


def detect_absence_phrases(text: str) -> tuple[str | None, str | None]:
    """Определяет по тексту статус отсутствия.

    Returns:
        (status, matched_text) где status = "away" | "soon_back" | None.
    """
    for pattern in _AWAY_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return ("away", text[:100])
    for pattern in _SOON_BACK_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return ("soon_back", text[:100])
    return (None, None)
