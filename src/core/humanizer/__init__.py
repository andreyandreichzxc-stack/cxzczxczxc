"""Humanizer — anti-AI детекция и очеловечивание текста."""

from .scorer import analyze_ai_score
from .humanizer import humanize_text, humanize_response
from .vocabulary import (
    AI_MARKERS,
    REPEAT_PENALTY,
    REPEAT_THRESHOLD,
    MAX_THEORETICAL_SCORE,
)
from .patterns import AI_PATTERNS, IDEAL_LENGTH_MIN, IDEAL_LENGTH_MAX
from .stats import record_check, get_stats

__all__ = [
    "analyze_ai_score",
    "humanize_text",
    "humanize_response",
    "AI_MARKERS",
    "REPEAT_PENALTY",
    "REPEAT_THRESHOLD",
    "MAX_THEORETICAL_SCORE",
    "AI_PATTERNS",
    "IDEAL_LENGTH_MIN",
    "IDEAL_LENGTH_MAX",
    "record_check",
    "get_stats",
]
