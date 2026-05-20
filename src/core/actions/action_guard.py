"""Guardrails for intent dispatch."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.core.actions.action_registry import action_registry
from src.core.intelligence.soul_snapshot import soul_snapshot


@dataclass
class GuardResult:
    allowed: bool
    intent: dict[str, Any]
    reason: str = ""
    risk_level: str = "low"


ALLOWED_SETTING_KEYS = {
    "auto_reply_enabled",
    "auto_reply_mode",
    "auto_reply_text",
    "auto_reply_cooldown_min",
    "digest_enabled",
    "digest_time",
    "news_enabled",
    "news_digest_time",
    "news_window_hours",
    "reminders_enabled",
    "reminder_lead_hours",
    "reminder_overdue_enabled",
    "ignore_archived",
    "use_heavy_model",
    "llm_provider",
    "transcription_mode",
    "transcription_api_provider",
    "auto_sync_enabled",
    "auto_sync_interval_sec",
    "auto_extract_memories",
    "include_saved_messages",
    "smart_digest_enabled",
    "smart_digest_interval_min",
    "urgent_notify_enabled",
    "monitor_only_selected_folders",
    "monitored_folders",
    "timezone",
}


def validate_intent_schema(intent: dict[str, Any]) -> GuardResult:
    if not isinstance(intent, dict):
        return GuardResult(False, {}, "Некорректный intent.")
    kind = str(intent.get("intent") or "")
    spec = action_registry.get(kind)
    if spec is None:
        return GuardResult(False, intent, f"Неизвестное действие: {kind or 'empty'}.")
    sanitized = action_registry.sanitize(intent)
    missing = [key for key in spec.required if not sanitized.get(key)]
    if missing:
        return GuardResult(
            False,
            sanitized,
            "Не хватает полей: " + ", ".join(sorted(missing)),
            spec.risk_level,
        )
    return GuardResult(True, sanitized, risk_level=spec.risk_level)


def guard_intent(intent: dict[str, Any]) -> GuardResult:
    schema = validate_intent_schema(intent)
    if not schema.allowed:
        return schema

    kind = str(schema.intent.get("intent") or "")
    if kind == "set_setting":
        key = str(schema.intent.get("key") or "")
        if key not in ALLOWED_SETTING_KEYS:
            return GuardResult(False, schema.intent, f"Настройка `{key}` не разрешена.", "high")

    if kind == "forget_memory" and not schema.intent.get("confirm_multi"):
        query = str(schema.intent.get("query") or "")
        broad = query.strip().lower() in {"all", "все", "всё", "*"} or len(query.strip()) < 3
        if broad:
            return GuardResult(False, schema.intent, "Слишком широкий запрос на удаление памяти.", "critical")

    tier = "context" if schema.risk_level in {"high", "critical"} else "volatile"
    try:
        serialized = json.dumps(schema.intent, ensure_ascii=False, sort_keys=True)
        allowed, reason = soul_snapshot.safety_gate(tier, serialized)
        if not allowed:
            return GuardResult(False, schema.intent, reason, schema.risk_level)
    except Exception:
        return GuardResult(False, schema.intent, "Safety gate failed.", schema.risk_level)

    return schema

