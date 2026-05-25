"""Pre-LLM gate — handles common patterns without calling LLM."""

from __future__ import annotations

import re

# Patterns that don't need LLM
_GREETINGS = {
    "привет",
    "здравствуй",
    "хай",
    "hello",
    "hi",
    "ку",
    "доброе утро",
    "добрый день",
    "добрый вечер",
    "доброй ночи",
    "здарова",
    "прив",
}
_FAREWELLS = {
    "пока",
    "до свидания",
    "спокойной ночи",
    "увидимся",
    "bye",
    "goodbye",
    "до завтра",
}
_AFFIRMATIVE = {
    "да",
    "ага",
    "угу",
    "ок",
    "окей",
    "ладно",
    "хорошо",
    "yes",
    "ok",
    "yep",
}
_NEGATIVE = {
    "нет",
    "не",
    "неа",
    "no",
    "nope",
}


def check_pre_gate(text: str) -> str | None:
    """Return a pre-canned response if text matches a known pattern, else None."""
    t = text.strip().lower().rstrip(".!?")

    # Greetings
    if t in _GREETINGS:
        return "Привет! Чем могу помочь?"

    # Farewells
    if t in _FAREWELLS:
        return "До связи! Если что — я здесь."

    # Pure affirmation — already handled by smart_reply emoji stage
    # but catch multi-word affirmatives that slip through
    if t in _AFFIRMATIVE:
        return None  # Already handled by Stage 0 emoji replies

    return None
