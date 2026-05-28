"""Intent classification — определяет режим ответа, риск, мгновенные ответы."""

from __future__ import annotations

import enum
import logging
import re as _re

from ..routing_wordlists import (
    HEAVY_WORDS,
    RISK_CRITICAL_WORDS,
    RISK_HIGH_WORDS,
    RISK_MEDIUM_WORDS,
    INSTANT_GREETING_PATTERN,
    INSTANT_HOWAREYOU_PATTERN,
    INSTANT_BYE_PATTERN,
    INSTANT_THANKS_PATTERN,
    INSTANT_ACK_PATTERN,
    INSTANT_REPLIES,
)

logger = logging.getLogger(__name__)


class RoutePurpose(str, enum.Enum):
    MAIN = "main"
    DRAFT = "draft"
    MEMORY = "memory"
    BACKGROUND = "background"
    SEARCH = "search"
    ANALYSIS = "analysis"
    URGENT = "urgent"
    FALLBACK = "fallback"


class RiskLevel(str, enum.Enum):
    LOW = "low"  # болтовня, привет
    MEDIUM = "medium"  # задача, поиск
    HIGH = "high"  # отправка сообщения, настройки
    CRITICAL = "critical"  # удаление данных, конфликты


class ResponseMode(str, enum.Enum):
    INSTANT = "instant"
    FAST_ROUTE = "fast_route"
    MAESTRO = "maestro"


# Learned routing: map intent_category → (purpose, risk, need_agents, heavy, cache_ttl)
_LEARNED_TASK_MAP: dict[str, tuple[str, str, tuple[str, ...], bool, int]] = {
    "send_message": ("main", "high", ("search", "draft"), False, 0),
    "draft_reply": ("draft", "medium", ("draft",), False, 0),
    "search": ("search", "medium", ("search",), True, 300),
    "find_in_chats": ("search", "medium", ("search",), True, 300),
    "summarize_chat": ("analysis", "medium", ("search",), True, 300),
    "tasks_for_chat": ("main", "medium", ("commitment",), False, 60),
    "catchup": ("main", "medium", ("search", "summarizer"), True, 180),
    "news_digest": ("analysis", "medium", ("search",), True, 300),
    "add_reminder": ("main", "medium", ("commitment",), False, 0),
    "store_memory": ("memory", "low", ("memory",), False, 0),
    "add_api_key": ("main", "high", (), False, 0),
    "remove_api_key": ("main", "critical", (), False, 0),
    "list_keys": ("main", "low", (), False, 60),
    "check_memories": ("memory", "low", ("memory",), False, 60),
    "set_setting": ("main", "high", (), False, 0),
}

INSTANT_PATTERNS = [
    _re.compile(INSTANT_GREETING_PATTERN),
    _re.compile(INSTANT_HOWAREYOU_PATTERN),
    _re.compile(INSTANT_BYE_PATTERN),
    _re.compile(INSTANT_THANKS_PATTERN),
    _re.compile(INSTANT_ACK_PATTERN),
]


async def classify_mode(user_text: str) -> ResponseMode:
    """Определяет режим ответа: instant / fast_route / maestro."""
    t = user_text.lower().strip()
    if len(t) < 30:
        for pattern in INSTANT_PATTERNS:
            m = _re.match(pattern, t)
            if m:
                return ResponseMode.INSTANT
    if len(t) < 100 and not any(w in t for w in HEAVY_WORDS):
        return ResponseMode.FAST_ROUTE
    return ResponseMode.MAESTRO


def get_instant_reply(user_text: str) -> str:
    """Возвращает мгновенный ответ для простых фраз."""
    t = user_text.lower().strip().rstrip(".!?")
    if t in INSTANT_REPLIES:
        return INSTANT_REPLIES[t]
    for key, reply in INSTANT_REPLIES.items():
        if t.startswith(key):
            return reply
    return ""


async def classify_risk(user_text: str) -> RiskLevel:
    """Быстрая эвристика для определения уровня риска."""
    t = user_text.lower().strip()
    # CRITICAL: удаление, сброс
    if any(w in t for w in RISK_CRITICAL_WORDS):
        return RiskLevel.CRITICAL
    # HIGH: отправка, настройки
    if any(w in t for w in RISK_HIGH_WORDS):
        return RiskLevel.HIGH
    # MEDIUM: поиск, анализ
    if any(w in t for w in RISK_MEDIUM_WORDS):
        return RiskLevel.MEDIUM
    return RiskLevel.LOW
