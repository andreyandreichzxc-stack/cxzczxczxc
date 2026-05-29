"""Анализатор AI-метрик текста. Pure function."""

import re
from collections import Counter

from .vocabulary import (
    get_ai_markers,
    MARKER_EXCEPTIONS,
    REPEAT_PENALTY,
    REPEAT_THRESHOLD,
    MAX_THEORETICAL_SCORE,
)
from .patterns import AI_PATTERNS, REPEAT_PENALTIES

# Pre-compiled emoji regex — one allocation at import, not per call
_EMOJI_RE = re.compile(
    r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    r"\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251"
    r"\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF"
    r"\U00002600-\U000026FF\U0001F7E0-\U0001F7FF]"
)


def analyze_ai_score(text: str, user_slots: list | None = None) -> tuple[float, dict]:
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

    # 1. Markers — собираем динамически из базовых + пользовательских моделей
    markers = get_ai_markers(user_slots)
    for phrase, weight in markers.items():
        if phrase.lower() not in text_lower:
            continue
        # Skip markers that match an exception phrase (case-insensitive)
        exceptions = MARKER_EXCEPTIONS.get(phrase, [])
        if any(exc in text_lower for exc in exceptions):
            continue
        score += weight
        if weight >= 0.3:
            breakdown["markers"].append({"phrase": phrase, "weight": weight})

    # 2. Patterns
    for pattern, weight, label in AI_PATTERNS:
        if pattern.search(text):
            score += weight
            breakdown["patterns"].append({"label": label, "weight": weight})

    # 3. Repeat check (uses per-word REPEAT_PENALTIES if available,
    #    otherwise falls back to global REPEAT_PENALTY)
    words = text_lower.split()
    word_counts = Counter(words)

    # 3a. Multi-word phrases — проверяем ОДИН раз до цикла по словам
    for phrase, phrase_penalty in REPEAT_PENALTIES.items():
        if " " not in phrase:
            continue
        occurrences = text_lower.count(phrase)
        if occurrences > REPEAT_THRESHOLD:
            extra = phrase_penalty * (occurrences - REPEAT_THRESHOLD)
            score += extra
            breakdown["repeats"].append(
                {
                    "word": phrase,
                    "count": occurrences,
                    "penalty": extra,
                }
            )

    # 3b. Single-word repeats — только однословные penalty
    for word, count in word_counts.items():
        if count > REPEAT_THRESHOLD and len(word) > 3:
            base_penalty = REPEAT_PENALTIES.get(word, REPEAT_PENALTY)
            penalty = base_penalty * (count - REPEAT_THRESHOLD)
            score += penalty
            if penalty > 0:
                breakdown["repeats"].append(
                    {"word": word, "count": count, "penalty": penalty}
                )

    # 4. Length check
    from .patterns import IDEAL_LENGTH_MIN  # noqa: F811

    if len(text) < IDEAL_LENGTH_MIN:
        score += 0.1
        breakdown["length_note"] = "слишком коротко"

    # Emoji density check
    emoji_count = len(_EMOJI_RE.findall(text))
    if emoji_count > 3:
        # Proportional penalty: больше эмодзи = выше penalty
        score += min(0.15 + max(0, emoji_count - 5) * 0.02, 0.35)

    # Normalize
    normalized = min(score / MAX_THEORETICAL_SCORE, 1.0)
    return normalized, breakdown
