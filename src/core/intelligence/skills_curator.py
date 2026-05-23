"""Skills Curator — lifecycle management for proposed skills.

Provides approval/rejection workflows, auto-approval of high-confidence
skills (confidence > 0.85 stored in YAML metadata), promotion to global
scope, and a background curation loop that runs every 6 hours.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from functools import partial
from typing import Any

from sqlalchemy import func, select

from src.config import settings
from src.core.infra.task_manager import task_manager
from src.core.scheduling.notification_queue import notification_queue
from src.db.models import Skill
from src.db.repo import get_or_create_user, get_skill_by_name, list_skills
from src.db.session import get_session

logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────


def _get_yaml_confidence(skill: Skill) -> float:
    """Extract confidence value from skill's YAML metadata.

    The YAML frontmatter is stored as ``{"__yaml__": {...}}`` inside
    ``trigger_patterns_json``.  Returns 0.0 if not found or unparseable.
    """
    patterns = skill.trigger_patterns_json or []
    for p in patterns:
        if isinstance(p, dict) and "__yaml__" in p:
            try:
                return float(p["__yaml__"].get("confidence", 0))
            except (TypeError, ValueError):
                return 0.0
    return 0.0


# ── curator API ──────────────────────────────────────────────────────


async def auto_approve_high_confidence() -> int:
    """Auto-approve proposed skills with confidence > 0.85.

    Scans all skills with ``review_status="proposed"`` for the current
    owner, checks their YAML metadata (``trigger_patterns_json["__yaml__"]``)
    for a ``confidence`` key, and approves those exceeding the 0.85
    threshold (sets ``review_status="approved"``, ``enabled=True``).

    Sends one notification to the owner summarising how many skills were
    approved.

    Returns:
        Number of skills approved.
    """
    async with get_session() as session:
        owner = await get_or_create_user(session, settings.owner_telegram_id)
        proposed = await list_skills(
            session, owner, review_status="proposed", limit=200
        )

        approved_count = 0
        for skill in proposed:
            confidence = _get_yaml_confidence(skill)
            if confidence > 0.85:
                skill.review_status = "approved"
                skill.enabled = True
                skill.updated_at = datetime.now(timezone.utc)
                approved_count += 1

        await session.flush()

    if approved_count:
        await notification_queue.enqueue(
            topic="skills",
            category="curator",
            priority=2,
            text=(
                f"🧠 Curator auto-approved {approved_count} skill(s) "
                f"with confidence > 85%."
            ),
        )
        logger.info("curator: auto-approved %d skills", approved_count)

    return approved_count


async def list_proposed() -> list[dict[str, Any]]:
    """Return all proposed skills sorted by confidence (descending).

    Each entry contains skill id, name, description, extracted confidence,
    and creation timestamp.
    """
    async with get_session() as session:
        owner = await get_or_create_user(session, settings.owner_telegram_id)
        skills = await list_skills(session, owner, review_status="proposed", limit=200)

    result: list[dict[str, Any]] = []
    for skill in skills:
        confidence = _get_yaml_confidence(skill)
        result.append(
            {
                "id": skill.id,
                "name": skill.name,
                "description": skill.description,
                "confidence": confidence,
                "created_at": skill.created_at.isoformat()
                if skill.created_at
                else None,
            }
        )

    result.sort(key=lambda x: x["confidence"], reverse=True)
    return result


async def approve_skill(owner_id: int, skill_name: str) -> bool:
    """Approve a proposed skill by name.

    Sets ``review_status="approved"`` and ``enabled=True``.

    Returns:
        True if the skill was found and updated, False otherwise.
    """
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        skill = await get_skill_by_name(session, owner, skill_name)
        if skill is None:
            logger.warning(
                "curator: approve_skill %r not found for %d",
                skill_name,
                owner_id,
            )
            return False
        skill.review_status = "approved"
        skill.enabled = True
        skill.updated_at = datetime.now(timezone.utc)
        await session.flush()

    logger.info("curator: approved skill %r (owner=%d)", skill_name, owner_id)
    return True


async def reject_skill(owner_id: int, skill_name: str, reason: str = "") -> bool:
    """Reject a proposed skill.

    Sets ``review_status="rejected"`` and ``enabled=False``.
    If a *reason* is provided, it is appended to the skill description.

    Returns:
        True if the skill was found and updated, False otherwise.
    """
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        skill = await get_skill_by_name(session, owner, skill_name)
        if skill is None:
            logger.warning(
                "curator: reject_skill %r not found for %d",
                skill_name,
                owner_id,
            )
            return False
        skill.review_status = "rejected"
        skill.enabled = False
        if reason:
            note = f"\n\n[Rejected: {reason}]"
            skill.description = (skill.description or "") + note
        skill.updated_at = datetime.now(timezone.utc)
        await session.flush()

    logger.info("curator: rejected skill %r (owner=%d)", skill_name, owner_id)
    return True


async def promote_to_global(owner_id: int, skill_name: str) -> bool:
    """Copy a user-scoped skill to global scope (``user_id=0``).

    A global skill is available to all users.  Only the original owner
    can promote; the original skill remains unchanged.

    If a global skill with the same name already exists, promotion is
    skipped.

    Returns:
        True if the skill was promoted, False if not found or already global.
    """
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        skill = await get_skill_by_name(session, owner, skill_name)
        if skill is None:
            logger.warning(
                "curator: promote_to_global %r not found for %d",
                skill_name,
                owner_id,
            )
            return False

        # Check if a global variant already exists
        existing = (
            await session.execute(
                select(Skill).where(
                    Skill.user_id == 0,
                    func.lower(Skill.name) == skill_name.lower().strip(),
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            logger.info(
                "curator: global skill %r already exists, skipping promote",
                skill_name,
            )
            return False

        global_skill = Skill(
            user_id=0,
            name=skill.name,
            description=skill.description,
            trigger_patterns_json=skill.trigger_patterns_json,
            body=skill.body,
            enabled=True,
            review_status="approved",
        )
        session.add(global_skill)
        await session.flush()

    logger.info(
        "curator: promoted skill %r from user %d to global",
        skill_name,
        owner_id,
    )
    return True


async def curator_stats(owner_id: int) -> dict[str, int]:
    """Return curator statistics for the given owner.

    Returns a dict with keys:
        proposed  — count of proposed skills
        approved  — count of approved skills
        rejected  — count of rejected skills
        global    — count of global (user_id=0) skills
        total     — sum of proposed + approved + rejected
    """
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)

        base = select(func.count(Skill.id)).where(Skill.user_id == owner.id)

        proposed_cnt = (
            await session.execute(base.where(Skill.review_status == "proposed"))
        ).scalar() or 0

        approved_cnt = (
            await session.execute(base.where(Skill.review_status == "approved"))
        ).scalar() or 0

        rejected_cnt = (
            await session.execute(base.where(Skill.review_status == "rejected"))
        ).scalar() or 0

        global_cnt = (
            await session.execute(
                select(func.count(Skill.id)).where(Skill.user_id == 0)
            )
        ).scalar() or 0

    return {
        "proposed": proposed_cnt,
        "approved": approved_cnt,
        "rejected": rejected_cnt,
        "global": global_cnt,
        "total": proposed_cnt + approved_cnt + rejected_cnt,
    }


# ── background loop ──────────────────────────────────────────────────


async def curator_loop(owner_telegram_id: int) -> None:
    """Background loop: every 6 hours run auto-approval + suggestions.

    Runs:
        1. ``auto_approve_high_confidence()``
        2. ``suggest_skills_from_trajectories(owner_telegram_id)``
        3. ``propose_skills_from_analysis(owner_telegram_id)``
    """
    from src.core.intelligence.skills import (
        propose_skills_from_analysis,
        suggest_skills_from_trajectories,
    )

    interval_sec = 6 * 3600  # 6 hours

    while True:
        try:
            approved = await auto_approve_high_confidence()
            if approved:
                logger.info("curator_loop: auto-approved %d skills", approved)
        except Exception:
            logger.exception("curator_loop: auto_approve_high_confidence failed")

        try:
            suggested = await suggest_skills_from_trajectories(owner_telegram_id)
            if suggested:
                await notification_queue.enqueue(
                    topic="skills",
                    category="curator",
                    priority=2,
                    text=(
                        f"🧠 Curator suggested {suggested} new skill(s) "
                        f"from recent trajectories."
                    ),
                )
        except Exception:
            logger.exception("curator_loop: suggest_skills_from_trajectories failed")

        try:
            proposed = await propose_skills_from_analysis(owner_telegram_id)
            if proposed:
                names = [s["name"] for s in proposed]
                await notification_queue.enqueue(
                    topic="skills",
                    category="curator",
                    priority=2,
                    text=(
                        f"🧠 Curator proposed {len(proposed)} skill(s) "
                        f"from analysis: {', '.join(names[:5])}."
                    ),
                )
        except Exception:
            logger.exception("curator_loop: propose_skills_from_analysis failed")

        await asyncio.sleep(interval_sec)


# ── task registration ────────────────────────────────────────────────

task_manager.register(
    "skill-curator",
    partial(curator_loop, settings.owner_telegram_id),
)
