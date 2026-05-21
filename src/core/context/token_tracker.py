"""Token budget tracker — approximate token counting for prompt management.

Uses a fast word-based heuristic (words × 1.3 ≈ tokens for Cyrillic/Latin).
Accurate enough for budget enforcement without the cost of a full tokenizer.
"""

from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)

# Budget thresholds (percentage of model max tokens)
BUDGET_MILD = 0.50  # light formatting at 50%
BUDGET_SUMMARY = 0.75  # LLM summarisation at 75%
BUDGET_AGGRESSIVE = 0.85  # Mermaid offload at 85%

# Default model max tokens (conservative — works for most models)
DEFAULT_MAX_TOKENS = 4096


def estimate_tokens(text: str) -> int:
    """Fast token count estimate: word count × 1.3.

    Works for Russian and English. Error margin ~10-15%.
    """
    if not text:
        return 0
    words = len(re.findall(r"\w+", text))
    return max(1, int(words * 1.3))


def count_prompt_tokens(
    system_prompt: str = "",
    user_prompt: str = "",
    history: list[dict] | None = None,
) -> int:
    """Estimate total tokens for an LLM prompt including history."""
    total = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
    if history:
        for msg in history:
            total += estimate_tokens(str(msg.get("content", "")))
            total += 4  # role + formatting overhead per message
    return total


def get_budget_stage(
    current_tokens: int,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[str, float]:
    """Determine which compression stage to apply.

    Returns (stage, fill_ratio).
    Stages: "ok" | "format" | "summary" | "mermaid"
    """
    ratio = current_tokens / max_tokens

    if ratio >= BUDGET_AGGRESSIVE:
        return ("mermaid", round(ratio, 2))
    if ratio >= BUDGET_SUMMARY:
        return ("summary", round(ratio, 2))
    if ratio >= BUDGET_MILD:
        return ("format", round(ratio, 2))
    return ("ok", round(ratio, 2))


__all__ = [
    "estimate_tokens",
    "count_prompt_tokens",
    "get_budget_stage",
    "DEFAULT_MAX_TOKENS",
    "BUDGET_MILD",
    "BUDGET_SUMMARY",
    "BUDGET_AGGRESSIVE",
]
