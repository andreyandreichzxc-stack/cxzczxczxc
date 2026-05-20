"""Adaptive context depth gates for fast memory recall."""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.core import conversation_context as ctx_store


RecallMode = str  # "light" | "normal" | "deep"


DEEP_HINTS = (
    "помнишь",
    "вспомни",
    "что было",
    "когда мы",
    "почему",
    "история",
    "раньше",
    "давно",
    "в марте",
    "в апреле",
    "в мае",
    "за неделю",
    "за месяц",
)

MEDIUM_HINTS = (
    "найди",
    "поиск",
    "сводка",
    "анализ",
    "напомни",
    "задач",
    "обещан",
    "отправь",
    "напиши",
    "скажи",
)


@dataclass(frozen=True)
class DepthDecision:
    depth: int
    message_weight: float
    recall_mode: RecallMode
    history_limit: int
    include_deep: bool


def get_depth(telegram_id: int) -> int:
    return ctx_store.get_recent_turn_count(telegram_id, max_age_seconds=3600)


def message_weight(text: str) -> float:
    t = (text or "").strip().lower()
    if not t:
        return 0.0
    word_count = len(re.findall(r"\w+", t, flags=re.UNICODE))
    weight = min(len(t) * 0.01, 2.5) + min(word_count * 0.08, 1.2)
    if any(hint in t for hint in DEEP_HINTS):
        weight += 2.0
    elif any(hint in t for hint in MEDIUM_HINTS):
        weight += 0.8
    if "?" in t:
        weight += 0.2
    return weight


def get_recall_mode(depth: int, weight: float, text: str = "") -> RecallMode:
    t = (text or "").lower()
    if any(hint in t for hint in DEEP_HINTS):
        return "deep"
    if weight < 0.5 and depth <= 2:
        return "light"
    if depth >= 9 or weight >= 2.0:
        return "deep"
    if depth >= 3 or weight >= 0.5:
        return "normal"
    return "light"


def decide_context_depth(telegram_id: int, text: str) -> DepthDecision:
    depth = get_depth(telegram_id)
    weight = message_weight(text)
    mode = get_recall_mode(depth, weight, text)
    if mode == "light":
        history_limit = 2
    elif mode == "normal":
        history_limit = 6
    else:
        history_limit = 8
    return DepthDecision(
        depth=depth,
        message_weight=weight,
        recall_mode=mode,
        history_limit=history_limit,
        include_deep=mode == "deep",
    )

