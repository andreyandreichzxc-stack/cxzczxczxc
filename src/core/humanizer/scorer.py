"""Анализатор AI-метрик текста. Pure function."""

from collections import Counter

from .vocabulary import (
    AI_MARKERS,
    REPEAT_PENALTY,
    REPEAT_THRESHOLD,
    MAX_THEORETICAL_SCORE,
)
from .patterns import AI_PATTERNS


def analyze_ai_score(text: str) -> tuple[float, dict]:
    """
    Возвращает (score, breakdown).
    score: 0.0 (человечно) .. 1.0 (AI-шаблон).
    breakdown: {"markers": [...], "patterns": [...], "repeats": [...]}
    """
    score = 0.0
    breakdown: dict = {"markers": [], "patterns": [], "repeats": [], "length_note": ""}
    if not text:
        return 0.02, breakdown  # empty/None = trivial, not AI
    text_lower = text.lower()

    # 1. Markers
    for phrase, weight in AI_MARKERS.items():
        if phrase.lower() in text_lower:
            score += weight
            if weight >= 0.3:
                breakdown["markers"].append({"phrase": phrase, "weight": weight})

    # 2. Patterns
    for pattern, weight, label in AI_PATTERNS:
        if pattern.search(text):
            score += weight
            breakdown["patterns"].append({"label": label, "weight": weight})

    # 3. Repeat check
    words = text_lower.split()
    word_counts = Counter(words)
    for word, count in word_counts.items():
        if count > REPEAT_THRESHOLD and len(word) > 3:
            penalty = REPEAT_PENALTY * (count - REPEAT_THRESHOLD)
            score += penalty
            breakdown["repeats"].append(
                {"word": word, "count": count, "penalty": penalty}
            )

    # 4. Length check
    from .patterns import IDEAL_LENGTH_MIN  # noqa: F811

    if len(text) < IDEAL_LENGTH_MIN:
        score += 0.1
        breakdown["length_note"] = "слишком коротко"

    # Normalize
    normalized = min(score / MAX_THEORETICAL_SCORE, 1.0)
    return normalized, breakdown
