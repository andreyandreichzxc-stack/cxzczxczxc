"""Guardrails: risk-based action approval module.

Decides whether an action needs user confirmation based on risk level
and conversational context.  Provides sanitisation, human-readable
confirmation messages, and a single evaluation entry point.

Intended for integration into maestro.py and free_text_pipeline.py
(Phase 2 — only the module is built here).
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any

from src.core.actions.action_registry import SAFE_KEYS, action_registry

logger = logging.getLogger(__name__)


# ── Risk level ─────────────────────────────────────────────────────────


class ActionRisk(enum.Enum):
    """How risky an action is from a safety standpoint.

    Levels progress from completely safe (no confirmation needed) to
    destructive (explicit user intent required).
    """

    LOW = "low"
    """No confirmation needed — read-only or trivial actions."""

    MEDIUM = "medium"
    """Confirm if the target context is new/unfamiliar (e.g. new contact)."""

    HIGH = "high"
    """Always confirm — actions that mutate state or affect others."""

    CRITICAL = "critical"
    """Require explicit intent — destructive irreversible actions."""


# ── Intent → risk mapping ─────────────────────────────────────────────

_INTENT_RISK_MAP: dict[str, ActionRisk] = {
    # LOW — read-only, informational, harmless
    "search": ActionRisk.LOW,
    "find_in_chats": ActionRisk.LOW,
    "show_inbox": ActionRisk.LOW,
    "show_digest": ActionRisk.LOW,
    "show_today": ActionRisk.LOW,
    "show_self": ActionRisk.LOW,
    "show_profile": ActionRisk.LOW,
    "show_style": ActionRisk.LOW,
    "show_skills": ActionRisk.LOW,
    "show_threads": ActionRisk.LOW,
    "show_trajectory": ActionRisk.LOW,
    "list_memories": ActionRisk.LOW,
    "list_todos": ActionRisk.LOW,
    "list_keys": ActionRisk.LOW,
    "chat": ActionRisk.LOW,
    "clarify": ActionRisk.LOW,
    "unknown": ActionRisk.LOW,
    "summarize_chat": ActionRisk.LOW,
    "catchup": ActionRisk.LOW,
    "tasks_for_chat": ActionRisk.LOW,
    "news_digest": ActionRisk.LOW,
    "check_memories": ActionRisk.LOW,
    # MEDIUM — state change but reversible or scoped
    "draft_reply": ActionRisk.MEDIUM,
    "set_reminder": ActionRisk.MEDIUM,
    "add_reminder": ActionRisk.MEDIUM,
    "remove_reminder": ActionRisk.MEDIUM,
    "add_reminders_from_chat": ActionRisk.MEDIUM,
    "add_news_topic": ActionRisk.MEDIUM,
    "remove_news_topic": ActionRisk.MEDIUM,
    "store_memory": ActionRisk.MEDIUM,
    "extract_memories_from_chat": ActionRisk.MEDIUM,
    "index_chats": ActionRisk.MEDIUM,
    "full_analysis": ActionRisk.MEDIUM,
    "change_auto_mode": ActionRisk.MEDIUM,
    "set_quiet_hours": ActionRisk.MEDIUM,
    "multi": ActionRisk.MEDIUM,
    "add_note": ActionRisk.MEDIUM,
    "change_settings": ActionRisk.MEDIUM,
    "suggest_reply": ActionRisk.MEDIUM,
    "summarize": ActionRisk.LOW,
    "show_contacts": ActionRisk.LOW,
    "list_news": ActionRisk.LOW,
    "/humanize": ActionRisk.LOW,
    # HIGH — state mutation, side effects, external communication
    "send_message": ActionRisk.HIGH,
    "send_draft": ActionRisk.HIGH,
    "delete_memory": ActionRisk.HIGH,
    "forget_memory": ActionRisk.HIGH,
    "add_contact": ActionRisk.HIGH,
    "delete_contact": ActionRisk.HIGH,
    "set_setting": ActionRisk.HIGH,
    "add_api_key": ActionRisk.HIGH,
    "toggle_api_key": ActionRisk.HIGH,
    "change_owner": ActionRisk.CRITICAL,
    "schedule_reminder": ActionRisk.HIGH,
    # CRITICAL — destructive, irreversible, or privilege-escalating
    "logout": ActionRisk.CRITICAL,
    "delete_data": ActionRisk.CRITICAL,
    "broadcast": ActionRisk.CRITICAL,
    "remove_api_key": ActionRisk.CRITICAL,
    "reset_all": ActionRisk.CRITICAL,
}


def get_action_risk(intent: str) -> ActionRisk:
    """Return the risk level for a given intent string.

    Unknown intents default to HIGH (safe side).
    """
    # Normalise: strip leading slashes, lowercase
    normalised = intent.strip().lstrip("/").lower()
    return _INTENT_RISK_MAP.get(normalised, ActionRisk.HIGH)


# ── Confirmation logic ────────────────────────────────────────────────


def needs_approval(intent: str, context: dict[str, Any] | None = None) -> bool:
    """Return True if the action requires user confirmation.

    ``context`` may contain overrides:
    - ``"skip_confirmation": True`` — skip approval for MEDIUM actions
      (used when the user explicitly said "do it without asking").
    - ``"is_new_contact": bool`` — for MEDIUM actions, confirm if True.
    - ``"bypass_reason": str`` — if ``"explicit_intent"`` is set, CRITICAL
      actions may still be allowed (caller must check separately).

    Risk-level rules:
    - LOW     → never confirm
    - MEDIUM  → confirm only when ``is_new_contact`` is True
    - HIGH    → always confirm
    - CRITICAL → always confirm (return True — caller should also
                 verify explicit user intent via ``bypass_reason``).
    """
    risk = get_action_risk(intent)
    ctx = context or {}

    if risk == ActionRisk.LOW:
        return False

    if risk == ActionRisk.MEDIUM:
        if ctx.get("skip_confirmation"):
            return False
        return bool(ctx.get("is_new_contact", False))

    # HIGH and CRITICAL always need approval
    if risk == ActionRisk.HIGH:
        return True

    # CRITICAL — never bypass, always require approval
    return True


# ── Confirmation message generation ───────────────────────────────────


def get_confirmation_message(intent: str, params: dict[str, Any]) -> str:
    """Generate a human-readable confirmation message in Russian.

    Example output:
      "Отправить сообщение Оле: «Привет, как дела?»?"
      "Удалить контакт «Иван Петров»?"
      "Забыть факт: «User works at Google»?"
    """
    risk = get_action_risk(intent)
    prefix = _RISK_PREFIX.get(risk, "")

    # Try intent-specific formatters first
    formatter = _CONFIRM_FORMATTERS.get(intent)
    if formatter is not None:
        msg = formatter(params)
        if msg:
            return f"{prefix}{msg}"

    # Generic fallback: describe from params
    desc = _describe_params(intent, params)
    return f"{prefix}{desc}"


_RISK_PREFIX: dict[ActionRisk, str] = {
    ActionRisk.HIGH: "⚠️ ",
    ActionRisk.CRITICAL: "🚨 ",
}


def _fmt_send_message(params: dict[str, Any]) -> str | None:
    recipient = params.get("recipient") or params.get("contact") or ""
    text = params.get("text", "")[:120]
    if recipient and text:
        return f"Отправить сообщение {recipient}: «{text}»?"
    if text:
        return f"Отправить сообщение: «{text}»?"
    return None


def _fmt_delete_memory(params: dict[str, Any]) -> str | None:
    query = params.get("query") or params.get("fact") or ""
    if query:
        return f"Забыть факт: «{query[:100]}»?"
    return "Удалить воспоминания?"


def _fmt_add_contact(params: dict[str, Any]) -> str | None:
    contact = params.get("contact") or params.get("contact_name") or ""
    if contact:
        return f"Добавить контакт «{contact}»?"
    return "Добавить новый контакт?"


def _fmt_delete_contact(params: dict[str, Any]) -> str | None:
    contact = params.get("contact") or params.get("contact_name") or ""
    peer = params.get("contact_id") or ""
    if contact:
        return f"Удалить контакт «{contact}»?"
    if peer:
        return f"Удалить контакт (id: {peer})?"
    return "Удалить контакт?"


def _fmt_set_setting(params: dict[str, Any]) -> str | None:
    key = params.get("key", "")
    value = params.get("value", "")
    if key:
        snippet = f"«{key}» = «{str(value)[:60]}»" if value else f"«{key}»"
        return f"Изменить настройку {snippet}?"
    return "Изменить настройки?"


def _fmt_add_api_key(params: dict[str, Any]) -> str | None:
    provider = params.get("provider", "")
    if provider:
        return f"Добавить API-ключ для {provider}?"
    return "Добавить API-ключ?"


def _fmt_remove_api_key(params: dict[str, Any]) -> str | None:
    slot = params.get("slot_id", "")
    if slot:
        return f"Удалить API-ключ (слот {slot})?"
    if params.get("all"):
        return "Удалить ВСЕ API-ключи?"
    return "Удалить API-ключ?"


def _fmt_logout(params: dict[str, Any]) -> str | None:
    return "Выйти из системы? Это остановит все активные сессии."


def _fmt_delete_data(params: dict[str, Any]) -> str | None:
    scope = params.get("scope", "all")
    return f"Удалить данные ({scope})? Это действие необратимо."


def _fmt_broadcast(params: dict[str, Any]) -> str | None:
    text = params.get("text", "")[:80]
    if text:
        return f"Отправить массовое сообщение: «{text}»?"
    return "Отправить массовое сообщение?"


def _fmt_forget_memory(params: dict[str, Any]) -> str | None:
    query = params.get("query", "")
    if query:
        return f"Удалить воспоминания по запросу «{query[:100]}»?"
    return "Удалить воспоминания?"


def _fmt_add_reminder(params: dict[str, Any]) -> str | None:
    text = params.get("text", "")[:80]
    when = params.get("when", "")
    if text and when:
        return f"Напомнить «{text}» в {when}?"
    if text:
        return f"Напомнить «{text}»?"
    return "Добавить напоминание?"


def _fmt_remove_reminder(params: dict[str, Any]) -> str | None:
    query = params.get("query", "")
    if query:
        return f"Удалить напоминание: «{query[:80]}»?"
    return "Удалить напоминание?"


def _fmt_change_auto_mode(params: dict[str, Any]) -> str | None:
    mode = params.get("mode", "")
    if mode:
        return f"Переключить режим авто-ответа на «{mode}»?"
    return "Изменить режим авто-ответа?"


def _fmt_set_quiet_hours(params: dict[str, Any]) -> str | None:
    start = params.get("start", "")
    end = params.get("end", "")
    if start and end:
        return f"Установить тихие часы с {start} до {end}?"
    return "Установить тихие часы?"


_CONFIRM_FORMATTERS: dict[str, Callable[..., str | None]] = {
    "send_message": _fmt_send_message,
    "send_draft": _fmt_send_message,
    "delete_memory": _fmt_delete_memory,
    "add_contact": _fmt_add_contact,
    "delete_contact": _fmt_delete_contact,
    "set_setting": _fmt_set_setting,
    "add_api_key": _fmt_add_api_key,
    "remove_api_key": _fmt_remove_api_key,
    "logout": _fmt_logout,
    "delete_data": _fmt_delete_data,
    "broadcast": _fmt_broadcast,
    "forget_memory": _fmt_forget_memory,
    "add_reminder": _fmt_add_reminder,
    "remove_reminder": _fmt_remove_reminder,
    "change_auto_mode": _fmt_change_auto_mode,
    "set_quiet_hours": _fmt_set_quiet_hours,
    "schedule_reminder": _fmt_add_reminder,
}


def _describe_params(intent: str, params: dict[str, Any]) -> str:
    """Generic fallback: build a short description from available params."""
    parts: list[str] = []
    for key in ("text", "query", "fact", "contact", "recipient", "topic", "question"):
        val = params.get(key)
        if val:
            s = str(val)[:80]
            parts.append(s)
            break
    if parts:
        return f"Действие «{intent}»: {parts[0]}?"
    return f"Выполнить «{intent}»?"


# ── Parameter sanitisation ────────────────────────────────────────────


def sanitize_action(intent: str, params: dict[str, Any]) -> dict[str, Any]:
    """Validate and sanitise action parameters.

    Uses the following strategy:
    1. If the intent is registered in ``action_registry``, keep only
       fields listed in its ``allowed`` set.
    2. Otherwise, keep only fields present in ``SAFE_KEYS``.
    3. Strip whitespace from string values.
    4. Remove empty / null values.
    """
    spec = action_registry.get(intent)
    allowed: set[str] = spec.allowed if spec is not None else SAFE_KEYS

    cleaned: dict[str, Any] = {}
    for k, v in params.items():
        if k not in allowed:
            continue
        if isinstance(v, str):
            v = v.strip()
            if not v:
                continue
        elif v is None:
            continue
        cleaned[k] = v

    # Always include intent
    cleaned["intent"] = intent
    return cleaned


# ── Evaluation result ─────────────────────────────────────────────────


@dataclass
class GuardrailResult:
    """Complete evaluation result for an action."""

    allowed: bool = True
    """Whether the action is permitted at all (always True for now;
    future versions may block CRITICAL actions outright)."""

    risk: ActionRisk = ActionRisk.LOW
    """Determined risk level."""

    needs_confirm: bool = False
    """Whether the action should go through user confirmation."""

    confirm_message: str = ""
    """Human-readable prompt to show the user (empty if no confirmation
    needed)."""

    sanitized_params: dict[str, Any] = field(default_factory=dict)
    """Cleaned parameter dict after sanitisation."""

    reason: str = ""
    """Optional explanation for the evaluation result."""


# ── Main entry point ──────────────────────────────────────────────────


def evaluate(
    intent: str,
    params: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> GuardrailResult:
    """Main entry point: run all guardrail checks and return a result.

    Steps:
    1. Determine risk level from intent name.
    2. Sanitise action parameters.
    3. Decide whether user confirmation is needed.
    4. Build a human-readable confirmation message if needed.

    Parameters
    ----------
    intent : str
        The action intent name (e.g. ``"send_message"``).
    params : dict
        Action parameters (e.g. ``{"recipient": "Оля", "text": "Привет"}``).
    context : dict or None
        Optional context overrides — see :func:`needs_approval`.

    Returns
    -------
    GuardrailResult
    """
    risk = get_action_risk(intent)
    sanitized = sanitize_action(intent, params)
    need_confirm = needs_approval(intent, context)

    reason_parts: list[str] = []
    reason_parts.append(f"risk={risk.value}")

    if need_confirm:
        confirm_msg = get_confirmation_message(intent, sanitized)
        reason_parts.append("needs_approval=True")
    else:
        confirm_msg = ""
        reason_parts.append("needs_approval=False")

    return GuardrailResult(
        allowed=True,
        risk=risk,
        needs_confirm=need_confirm,
        confirm_message=confirm_msg,
        sanitized_params=sanitized,
        reason="; ".join(reason_parts),
    )
