"""Dream cycle — unified nightly memory maintenance.

Replaces four separate background tasks with a single orchestrated job:
  1. Decay + tier promotion/demotion (was memory_checker @ 03:00)
  2. Duplicate consolidation (was memory_consolidator @ every 6h)
  3. Contradiction detection (was ad-hoc, per-message)
  4. Digest rebuild for top 20 active contacts (was on-access)

Runs once per day at 03:00 UTC and sends a single summary notification.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from src.config import settings
from src.db.session import get_session
from src.db.repo import get_or_create_user

logger = logging.getLogger(__name__)


async def dream_cycle(owner_telegram_id: int) -> None:
    """Run complete nightly memory maintenance.

    Executes all four phases sequentially.  Each phase is wrapped in its
    own try/except so a failure in one phase does not block the others.

    Phases:
        1. Decay + tier promotion/demotion
        2. Duplicate consolidation
        3. Contradiction detection (placeholder)
        4. Digest rebuild for active contacts
    """
    summary = {
        "decayed": 0,
        "closed": 0,
        "consolidated": 0,
        "contradictions": 0,
        "digests": 0,
    }

    # ── Phase 1: Decay + tier promotion/demotion ──────────────────
    try:
        from src.core.memory.memory_checker import _run_decay_and_validation

        decayed, closed = await _run_decay_and_validation(owner_telegram_id)
        summary["decayed"] = decayed
        summary["closed"] = closed
        logger.info(
            "Dream cycle: phase 1 (decay) — %d decayed, %d closed",
            decayed,
            closed,
        )
    except Exception:
        logger.exception("Dream cycle: phase 1 (decay) failed")

    # ── Phase 2: Duplicate consolidation ──────────────────────────
    try:
        from src.core.memory.memory_consolidator import consolidate_memories

        merged = await consolidate_memories(owner_telegram_id)
        summary["consolidated"] = merged
        logger.info(
            "Dream cycle: phase 2 (consolidation) — %d merged",
            merged,
        )
    except Exception:
        logger.exception("Dream cycle: phase 2 (consolidation) failed")

    # ── Phase 3: Contradiction batch scan ──────────────────────────
    try:
        from src.core.memory.contradiction_detector import _scan_contradictions_batch
        from src.db.repo import list_memories, get_or_create_user

        async with get_session() as session:
            owner = await get_or_create_user(session, owner_telegram_id)
            memories = await list_memories(session, owner, limit=200)
            contradictions = await _scan_contradictions_batch(
                memories, owner_telegram_id
            )
        summary["contradictions"] = contradictions
        logger.info("Dream cycle: phase 3 (contradictions) — %d found", contradictions)
    except Exception:
        logger.exception("Dream cycle: phase 3 (contradictions) failed")
        summary["contradictions"] = 0

    # ── Phase 4: Digest rebuild for top 20 active contacts ────────
    try:
        async with get_session() as session:
            owner = await get_or_create_user(session, owner_telegram_id)

            from src.db.repo import list_contacts

            contacts = await list_contacts(session, owner, include_bots=False)
            # Non-bot, active (peer_id > 0), top 20
            active = [c for c in contacts if c.peer_id > 0][:20]

            from src.core.contacts.contact_memory_digest import get_contact_digest

            for contact in active:
                try:
                    await get_contact_digest(owner.telegram_id, contact.peer_id)
                    summary["digests"] += 1
                except Exception:
                    pass

        logger.info(
            "Dream cycle: phase 4 (digests) — %d rebuilt",
            summary["digests"],
        )

        # Also cleanup old conversation summaries (>7 days)
        try:
            from src.core.memory.conversation_context import cleanup_old_summaries

            await cleanup_old_summaries()
            logger.info("Dream cycle: cleaned up old conversation summaries")
        except Exception:
            pass
    except Exception:
        logger.exception("Dream cycle: phase 4 (digests) failed")

    # ── Summary notification ──────────────────────────────────────
    try:
        from src.core.scheduling.notification_queue import notification_queue

        await notification_queue.enqueue(
            topic="system",
            text=(
                "🌙 <b>Ночной цикл завершён</b>\n"
                f"• decay: {summary['decayed']} фактов обновлено, "
                f"{summary['closed']} закрыто\n"
                f"• консолидация: {summary['consolidated']} дубликатов смержено\n"
                f"• дайджесты: {summary['digests']} контактов перестроено"
            ),
            priority=3,  # PRIORITY_LOW — информационное
        )
    except Exception:
        pass


async def dream_loop(owner_telegram_id: int) -> None:
    """Run dream cycle once per day at 03:00 UTC.

    Calculates sleep duration to the next 03:00 target, executes the
    cycle, then repeats.  On fatal error sleeps 1 hour before retry.
    """
    while True:
        now = datetime.now(timezone.utc)
        # Calculate seconds until next 03:00 UTC
        next_run = now.replace(hour=3, minute=0, second=0, microsecond=0)
        if next_run <= now:
            next_run = next_run + timedelta(days=1)
        wait_sec = (next_run - now).total_seconds()

        logger.info(
            "Dream cycle: sleeping %.0f seconds until %s",
            wait_sec,
            next_run.isoformat(),
        )
        await asyncio.sleep(wait_sec)

        try:
            await dream_cycle(owner_telegram_id)
        except Exception:
            logger.exception("Dream cycle: fatal error, retrying in 1 hour")
            await asyncio.sleep(3600)  # retry in 1 hour


# ── Auto-register with task manager on import ────────────────────
from functools import partial
from src.core.infra.task_manager import task_manager

task_manager.register(
    "dream-cycle",
    partial(dream_loop, settings.owner_telegram_id),
    restart_on_failure=True,
    restart_delay=60,
)
