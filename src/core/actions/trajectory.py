"""Trajectory recording for the Hermes-like learning loop."""

from __future__ import annotations

import logging
from typing import Any

from src.db.repo import add_trajectory, get_or_create_user
from src.db.session import get_session

logger = logging.getLogger(__name__)


async def record_trajectory(
    telegram_id: int,
    *,
    request_text: str,
    route_mode: str | None = None,
    intent_json: dict | None = None,
    actions_json: list | None = None,
    used_skills_json: list | None = None,
    memory_ids_json: list | None = None,
    response_text: str | None = None,
    success: bool = True,
    error: str | None = None,
    latency_ms: int | None = None,
) -> int | None:
    """Best-effort trajectory write. Never breaks the user-facing turn."""
    try:
        async with get_session() as session:
            owner = await get_or_create_user(session, telegram_id)
            row = await add_trajectory(
                session,
                owner,
                request_text=request_text,
                route_mode=route_mode,
                intent_json=intent_json,
                actions_json=actions_json,
                used_skills_json=used_skills_json,
                memory_ids_json=memory_ids_json,
                response_text=response_text,
                success=success,
                error=error,
                latency_ms=latency_ms,
            )

            # Feature 2: Pattern cache recording — только при успехе и opt-in
            if success and route_mode:
                try:
                    if owner.settings and owner.settings.pattern_caching_enabled:
                        from src.core.intelligence.pattern_cache import pattern_cache

                        intent_type = (
                            str(intent_json.get("intent"))
                            if isinstance(intent_json, dict)
                            and intent_json.get("intent")
                            else route_mode
                        )
                        action = route_mode
                        if isinstance(actions_json, list) and actions_json:
                            first_action = actions_json[0]
                            if isinstance(first_action, dict):
                                action = first_action.get("action", route_mode)
                        await pattern_cache.record_pattern(
                            telegram_id, intent_type, action
                        )
                except Exception:
                    logger.debug("Pattern cache recording skipped", exc_info=True)

            # Feature 3: Record skill usage telemetry
            if used_skills_json and row.id:
                try:
                    from src.core.intelligence.skills import record_skill_usages

                    await record_skill_usages(
                        telegram_id, used_skills_json, row.id, success
                    )
                except Exception:
                    logger.debug("Skill usage recording skipped", exc_info=True)

            return row.id
    except Exception:
        logger.debug("Failed to record trajectory", exc_info=True)
        return None


def actions_from_intent(intent: dict[str, Any] | None) -> list:
    if not isinstance(intent, dict):
        return []
    if intent.get("intent") == "multi" and isinstance(intent.get("actions"), list):
        return intent["actions"]
    return [intent]
