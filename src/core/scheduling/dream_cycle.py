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
        "dsm": 0,
        "auto_forgotten": 0,
        "stale_closed": 0,
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
                memories,
                owner_telegram_id,
                session=session,
                owner=owner,
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

    # ── Phase 5: Memory Wiki ───────────────────────────────────────
    try:
        from src.core.memory.memory_wiki import generate_memory_wiki

        wiki_stats = await generate_memory_wiki(owner_telegram_id)
        total_facts = sum(wiki_stats.values())
        logger.info(
            "Dream cycle: wiki generated (%d categories, %d facts)",
            len(wiki_stats),
            total_facts,
        )
    except Exception:
        logger.warning("Dream cycle: wiki generation failed", exc_info=True)

    # ── Phase 6: DSM cleanup ───────────────────────────────────────
    try:
        from src.core.intelligence.dsm import dsm_cleanup

        removed = await dsm_cleanup(days=30)
        summary["dsm"] = removed
        if removed:
            logger.info(
                "Dream cycle: phase 6 (DSM cleanup) — removed %d old entries", removed
            )
    except Exception:
        logger.exception("Dream cycle: phase 6 (DSM cleanup) failed")

    # ── Phase 7: Auto-forget sweep ─────────────────────────────────────
    try:
        from src.core.memory.auto_forget import auto_forget_sweep
        from src.db.repo import get_or_create_user

        async with get_session() as session:
            owner = await get_or_create_user(session, owner_telegram_id)
            forgotten = await auto_forget_sweep(session, owner.id)
            if forgotten:
                await session.commit()
        summary["auto_forgotten"] = forgotten
        if forgotten:
            logger.info(
                "Dream cycle: phase 7 (auto-forget) — %d facts deactivated",
                forgotten,
            )
    except Exception:
        logger.exception("Dream cycle: phase 7 (auto-forget) failed")
        summary["auto_forgotten"] = 0

    # ── Phase 8: Close stale sessions ──────────────────────────────────
    try:
        from src.core.memory.session_recorder import close_stale_sessions

        async with get_session() as session:
            stale_closed = await close_stale_sessions(session, max_age_hours=24)
        summary["stale_closed"] = stale_closed
        if stale_closed:
            logger.info(
                "Dream cycle: phase 8 (stale sessions) — %d closed",
                stale_closed,
            )
    except Exception:
        logger.exception("Dream cycle: phase 8 (stale sessions) failed")
        summary["stale_closed"] = 0

    # ── Graph statistics ──────────────────────────────────────────────
    try:
        from src.db.repos.memory_repo import get_graph_stats
        from src.db.repo import get_or_create_user

        async with get_session() as session:
            owner = await get_or_create_user(session, owner_telegram_id)
            graph_stats = await get_graph_stats(session, owner.id)
    except Exception:
        logger.exception("Dream cycle: graph stats failed")
        graph_stats = None

    # ── Retention statistics ──────────────────────────────────────────
    try:
        from src.core.memory.temporal_layers import compute_retention, utcnow_naive
        from src.db.repo import get_or_create_user, list_memories

        async with get_session() as session:
            owner = await get_or_create_user(session, owner_telegram_id)
            memories = await list_memories(session, owner, is_active=True)
            now = utcnow_naive()
            retention_buckets = {"strong": 0, "fading": 0, "weak": 0}
            for m in memories:
                retention = compute_retention(m, now)
                if retention >= 0.8:
                    retention_buckets["strong"] += 1
                elif retention >= 0.5:
                    retention_buckets["fading"] += 1
                else:
                    retention_buckets["weak"] += 1
    except Exception:
        logger.exception("Dream cycle: retention stats failed")
        retention_buckets = None

    # ── Summary notification ──────────────────────────────────────
    try:
        from src.core.scheduling.notification_queue import notification_queue

        # — Build graph stats line —
        if graph_stats:
            gs = graph_stats
            ebt = gs.get("edges_by_type", {})
            supports = ebt.get("supports", 0)
            contradicts = ebt.get("contradicts", 0)
            related = ebt.get("related", 0)
            graph_line = (
                f"📊 Граф: {gs['node_count']} узлов, "
                f"{gs['total_edges']} рёбер "
                f"(supports: {supports}, contradicts: {contradicts}, "
                f"related: {related})\n"
            )
        else:
            graph_line = ""

        # — Build retention stats line —
        if retention_buckets:
            rb = retention_buckets
            retention_line = (
                f"• удержание: 🔒 strong {rb['strong']}, "
                f"⏳ fading {rb['fading']}, "
                f"📦 weak {rb['weak']}\n"
            )
        else:
            retention_line = ""

        # — Build optional lines (only shown when > 0) —
        stale_line = (
            f"• закрыто сессий: {summary['stale_closed']}\n"
            if summary.get("stale_closed")
            else ""
        )
        forgotten_line = (
            f"• авто-забывание: {summary['auto_forgotten']} фактов деактивировано\n"
            if summary.get("auto_forgotten")
            else ""
        )

        await notification_queue.enqueue(
            topic="system",
            text=(
                "🌙 <b>Ночной цикл завершён</b>\n"
                f"• decay: {summary['decayed']} фактов обновлено, "
                f"{summary['closed']} закрыто\n"
                f"• консолидация: {summary['consolidated']} дубликатов смержено\n"
                f"• дайджесты: {summary['digests']} контактов перестроено\n"
                f"{stale_line}"
                f"{forgotten_line}"
                f"{retention_line}"
                f"{graph_line}"
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
