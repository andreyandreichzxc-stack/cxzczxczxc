"""Correction Learner — learns from user corrections to improve over time.

When the user corrects the bot ("нет, не так", "не, через два", "отмени"),
this module:
  1. Records the correction pattern in in-memory history
  2. Feeds it into the humanizer feedback loop (to improve future rewrites)
  3. Updates adaptive persona if it's a style correction
  4. Updates memory if it's a fact correction

Integration point: smart_correction stage 0d → learn_correction()
Context injection:  maestro pre-loads recent corrections → prompt_assembler injects.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.db.session import get_session
from src.db.repo import get_or_create_user, get_persona, update_persona

logger = logging.getLogger(__name__)

# ── In-memory correction history ──────────────────────────────────────
# Structure: {telegram_id: [(original_text, corrected_text, timestamp), ...]}
_correction_history: dict[int, list[tuple[str, str, float]]] = {}
_correction_lock = asyncio.Lock()
_MAX_HISTORY = 50  # max entries per user


async def learn_correction(
    telegram_id: int,
    original_text: str,
    corrected_text: str,
    feedback_type: str = "rewrite",  # "rewrite" | "fact" | "style" | "cancel"
) -> None:
    """Record a correction for future learning.

    Args:
        telegram_id:  Telegram user ID.
        original_text: What the bot said/did (or the raw user command).
        corrected_text: What the user wanted instead.
        feedback_type:  Category of correction.
    """
    # ── 1. Store in in-memory history ──
    async with _correction_lock:
        if telegram_id not in _correction_history:
            _correction_history[telegram_id] = []
        history = _correction_history[telegram_id]
        history.append(
            (
                original_text[:500],
                corrected_text[:500],
                asyncio.get_event_loop().time(),
            )
        )
        if len(history) > _MAX_HISTORY:
            history.pop(0)  # evict oldest

    logger.debug(
        "Correction learned: type=%s, user=%d, %d history entries",
        feedback_type,
        telegram_id,
        len(history),
    )

    # ── 2. Feed into humanizer feedback loop ──
    try:
        from src.core.humanizer.humanizer import record_humanizer_feedback

        record_humanizer_feedback(
            telegram_id,
            original=original_text,
            corrected=corrected_text,
            accepted=False,
        )
    except Exception:
        pass  # humanizer feedback is optional

    # ── 3. Update adaptive persona if style correction ──
    if feedback_type == "style" and corrected_text:
        try:
            async with get_session() as session:
                owner = await get_or_create_user(session, telegram_id)
                persona = await get_persona(session, owner)

                from src.core.intelligence.adaptive_persona import (
                    detect_persona_change,
                )

                change = await detect_persona_change(corrected_text)
                if change and change.get("changes"):
                    await update_persona(
                        session,
                        persona,
                        **change["changes"],
                        auto=True,  # accepted by maintainability lint
                    )
                    logger.info(
                        "Persona updated from correction for user %d: %s",
                        telegram_id,
                        change.get("reason", "unknown"),
                    )
        except Exception:
            logger.debug("Persona update from correction failed", exc_info=True)

    # ── 4. If fact correction — queue memory update ──
    if feedback_type == "fact" and corrected_text:
        try:
            from src.core.memory.memory_queue import enqueue, MemoryJob

            await enqueue(
                MemoryJob(
                    telegram_id=telegram_id,
                    facts=[{"fact": corrected_text, "confidence": 0.9}],
                    job_type="save",
                )
            )
        except asyncio.QueueFull:
            logger.debug(
                "Memory queue full, fact correction dropped for user %d", telegram_id
            )
        except Exception:
            logger.debug("Memory update from correction failed", exc_info=True)


async def get_recent_corrections(
    telegram_id: int,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Get recent corrections for context injection into system prompt.

    Returns:
        List of dicts with keys: original, corrected.
    """
    async with _correction_lock:
        history = _correction_history.get(telegram_id, [])
        return [
            {"original": orig, "corrected": corr} for orig, corr, _ in history[-limit:]
        ]


async def get_correction_stats(telegram_id: int) -> dict[str, int]:
    """Return correction statistics for /health or dashboards."""
    async with _correction_lock:
        return {
            "user_corrections": len(_correction_history.get(telegram_id, [])),
            "global_total": sum(len(v) for v in _correction_history.values()),
        }
