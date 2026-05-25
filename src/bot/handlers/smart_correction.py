"""Smart correction — detects when user corrects/cancels their own command.

Examples:
  «напомни через час» → «нет, через два»  — REPLACEMENT of previous
  «не насте, а маше напиши»               — corrects recipient in current draft
  «отмени» / «передумал»                  — cancels previous action
"""

from __future__ import annotations

import asyncio
import logging
import re
import time

logger = logging.getLogger(__name__)

# Track last action per user for correction context
_last_actions: dict[int, dict] = {}
_actions_lock = asyncio.Lock()

# Time-to-live for recorded actions: corrections must come within 60 seconds
_ACTION_TTL = 60.0


# ── Correction pattern matching ─────────────────────────────────────


# "нет, через два" / "не так, в среду" → replaces parameter of previous
_CORRECTION_NO_RE = re.compile(
    r"^(?:нет|не|не так|ошибся|передумал)[,.]?\s+(.+)$",
    re.IGNORECASE,
)

# Bare "нет" / "не так" / "ошибся" (no arguments) → cancel
_CORRECTION_BARE_NO_RE = re.compile(
    r"^(?:нет|не так|ошибся|передумал)[,.]?\s*$",
    re.IGNORECASE,
)

# Explicit cancel: "отмени" / "отбой" / "не надо"
_CANCEL_RE = re.compile(
    r"^(?:отмени|отмена|передумал|не надо|отбой)\b",
    re.IGNORECASE,
)

# Replacement: "вместо X — Y" / "замени X на Y"
_REPLACE_RE = re.compile(
    r"(?:вместо|замени|заменить)\s+(.+?)\s+(?:—|—|-|→|->|на)\s+(.+)",
    re.IGNORECASE,
)

# Time extraction: "через X часов/минут/дней"
_TIME_RE = re.compile(
    r"(?:через|в)\s+(\d+)\s*(?:час|ч|мин|минут|день|дн|дней|недел[ьюи])",
    re.IGNORECASE,
)

# Contact detection: "контакту X" / "для X" in correction text
_CONTACT_RE = re.compile(
    r"(?:для|контакту?)\s+(\S+)",
    re.IGNORECASE,
)


async def record_action(user_id: int, action: dict) -> None:
    """Record a user action for potential correction.

    Args:
        user_id: Telegram user ID
        action: dict with 'intent' (str) and 'params' (dict)
    """
    async with _actions_lock:
        _last_actions[user_id] = {
            "intent": action.get("intent", ""),
            "params": action.get("params", {}),
            "timestamp": time.monotonic(),
        }


async def detect_correction(user_id: int, text: str) -> dict | None:
    """Detect if text is a correction of a previous action.

    Returns:
        None if no correction detected.
        dict with:
          - action: "cancel" | "replace"
          - previous: the recorded action dict
          - new_params: (replace only) merged params
          - new_text: (replace only) the raw new text after prefix
    """
    text_stripped = text.strip()

    # ── Explicit cancel: «отмени», «отбой», «не надо» ───────────────
    if _CANCEL_RE.match(text_stripped):
        async with _actions_lock:
            prev = _pop_if_fresh(user_id)
        if prev:
            return {"action": "cancel", "previous": prev}
        return None

    # ── Bare «нет» / «не так» → cancel last action ──────────────────
    if _CORRECTION_BARE_NO_RE.match(text_stripped):
        async with _actions_lock:
            prev = _pop_if_fresh(user_id)
        if prev:
            return {"action": "cancel", "previous": prev}
        return None

    # ── «нет, ...» / «не, ...» → replace params ─────────────────────
    match = _CORRECTION_NO_RE.match(text_stripped)
    if match:
        async with _actions_lock:
            prev = _last_actions.get(user_id)
        if prev is None:
            return None
        if not _is_fresh(prev):
            async with _actions_lock:
                _last_actions.pop(user_id, None)
            return None

        new_text = match.group(1).strip()

        # Extract time from correction text
        time_match = _TIME_RE.search(new_text)
        # Extract contact from correction text
        contact_match = _CONTACT_RE.search(new_text)

        new_params = dict(prev.get("params", {}))
        if time_match:
            new_params["time"] = new_text
        if contact_match:
            new_params["contact"] = contact_match.group(1)

        # Remove the old record — correction consumed it
        async with _actions_lock:
            _last_actions.pop(user_id, None)

        return {
            "action": "replace",
            "previous": prev,
            "new_params": new_params,
            "new_text": new_text,
        }

    return None


async def apply_correction(user_id: int, correction: dict) -> str:
    """Apply a correction (cancel/update previous action).

    Returns:
        Confirmation message to show to user.
    """
    action = correction.get("action", "")
    prev = correction.get("previous", {})
    prev_intent = prev.get("intent", "предыдущее")

    if action == "cancel":
        return f"✅ Отменил «{prev_intent}»."

    if action == "replace":
        new_text = correction.get("new_text", "")
        return f"✅ Понял, обновил: {new_text}"

    return "✅ Готово."


# ── Internal helpers ────────────────────────────────────────────────


def _is_fresh(entry: dict) -> bool:
    """Check if the recorded action is still fresh (within TTL)."""
    return (time.monotonic() - entry.get("timestamp", 0)) < _ACTION_TTL


def _pop_if_fresh(user_id: int) -> dict | None:
    """Pop and return the recorded action if it's still fresh, else None."""
    entry = _last_actions.get(user_id)
    if entry is None:
        return None
    if not _is_fresh(entry):
        _last_actions.pop(user_id, None)
        return None
    return _last_actions.pop(user_id, None)
