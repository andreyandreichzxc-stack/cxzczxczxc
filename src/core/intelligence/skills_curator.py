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

# ── Rate limiting for skill edits ──
# Key: (owner_id, skill_name_lower), Value: last edit timestamp
_edit_cooldowns: dict[tuple[int, str], datetime] = {}
EDIT_COOLDOWN_SECONDS = 60  # Minimum 60 seconds between edits to the same skill
_COOLDOWN_TTL_SECONDS = 3600  # Evict entries older than 1 hour
MIN_USAGE_FOR_CALIBRATION = 5  # less than 5 uses → raw confidence only


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


def _calibrate_confidence(skill) -> float:
    """Калибрует LLM-confidence на основе реального success_rate.

    Если usage_count < MIN_USAGE_FOR_CALIBRATION — возвращает raw confidence
    (недостаточно данных для калибровки).
    Если usage_count >= MIN_USAGE_FOR_CALIBRATION — смешивает raw confidence
    с success_rate.  Чем больше usage — тем больше вес success_rate.
    """
    raw = _get_yaml_confidence(skill)

    usage_count = (skill.success_count or 0) + (skill.failure_count or 0)
    if usage_count < MIN_USAGE_FOR_CALIBRATION:
        logger.debug(
            "Skill '%s' has %d uses (<%d), using raw confidence %.3f",
            skill.name,
            usage_count,
            MIN_USAGE_FOR_CALIBRATION,
            raw,
        )
        return raw

    success_rate = (skill.success_count or 0) / max(usage_count, 1)

    usage_weight = min(0.9, 0.3 + 0.03 * min(usage_count, 20))

    calibrated = raw * (1 - usage_weight) + success_rate * usage_weight
    logger.debug(
        "Skill '%s': raw=%.2f × (1-%.2f) + success_rate=%.2f × %.2f = %.3f",
        skill.name,
        raw,
        usage_weight,
        success_rate,
        usage_weight,
        calibrated,
    )
    return round(calibrated, 3)


# ── curator API ──────────────────────────────────────────────────────


async def auto_approve_high_confidence() -> int:
    """Auto-approve proposed skills with confidence > 0.85.

    V2: Now includes validation gate — skills are validated against held-out
    trajectories before approval. Only skills that pass validation are approved.

    Scans all skills with ``review_status="proposed"`` for the current
    owner, checks their YAML metadata (``trigger_patterns_json["__yaml__"]``)
    for a ``confidence`` key, and approves those exceeding the 0.85
    threshold (sets ``review_status="approved"``, ``enabled=True``).

    Sends one notification to the owner summarising how many skills were
    approved.

    Returns:
        Number of skills approved.
    """
    from src.config import settings
    from src.core.intelligence.skill_validator import validate_skill_candidate

    async with get_session() as session:
        owner = await get_or_create_user(session, settings.owner_telegram_id)
        proposed = await list_skills(
            session, owner, review_status="proposed", limit=200
        )

        approved_count = 0
        rejected_count = 0
        for skill in proposed:
            confidence = _calibrate_confidence(skill)
            logger.debug(
                f"Skill {skill.name}: raw={_get_yaml_confidence(skill):.2f}, calibrated={confidence:.2f}"
            )
            if confidence > 0.85:
                # V2: Validation gate — validate before approval
                if settings.skill_validation_enabled:
                    validation = await validate_skill_candidate(
                        owner.id,
                        skill.name,
                        skill.body,
                    )
                    if not validation.accepted:
                        skill.review_status = "rejected"
                        skill.enabled = False
                        note = (
                            f"\n\n[Rejected by validation gate: {validation.reason}. "
                            f"Score: {validation.score_before:.2f} → {validation.score_after:.2f}]"
                        )
                        skill.description = (skill.description or "") + note
                        rejected_count += 1
                        logger.info(
                            "curator: validation rejected %r — %s",
                            skill.name,
                            validation.summary,
                        )
                        continue

                    skill.validation_score = validation.score_after
                    if validation.score_delta > 0:
                        skill.best_body = skill.body

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
                + (
                    f" Rejected {rejected_count} by validation gate."
                    if rejected_count
                    else ""
                )
            ),
        )
        logger.info(
            "curator: auto-approved %d skills, rejected %d by validation",
            approved_count,
            rejected_count,
        )

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

    V2: Also saves the rejection to the rejected-edits buffer for negative
    feedback in future optimization cycles.

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

            # V2: Save rejection to rejected-edits buffer
            rejected = skill.rejected_edits_json or []
            rejected.append(
                {
                    "op": "create",
                    "target": skill_name,
                    "content": (skill.body or "")[:200],
                    "reason": reason,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            # Keep only last 10 rejections
            skill.rejected_edits_json = rejected[-10:]

        skill.updated_at = datetime.now(timezone.utc)
        await session.flush()

    logger.info("curator: rejected skill %r (owner=%d)", skill_name, owner_id)
    return True


async def apply_skill_edit(
    owner_id: int,
    skill_name: str,
    edit_op: str,
    edit_target: str | None = None,
    edit_content: str = "",
    edit_reason: str = "",
    *,
    skip_validation: bool = False,
) -> dict:
    """Apply a bounded edit to an existing skill.

    Instead of replacing the entire skill body, applies a minimal targeted edit
    (append, insert_after, replace, delete) with edit budget enforcement.

    V2: SkillOpt-inspired bounded edits with validation gate.

    Args:
        owner_id: Owner user ID
        skill_name: Skill name to edit
        edit_op: Operation type (append/insert_after/replace/delete)
        edit_target: Target marker for insert_after, old text for replace/delete
        edit_content: New content for the edit
        edit_reason: Why this edit is proposed
        skip_validation: If True, skip validation gate (for manual edits)

    Returns:
        Dict with success, new_version, applied_edits, rejected_edits, validation
    """
    from src.config import settings
    from src.core.intelligence.skill_editor import (
        EditOp,
        SkillEdit,
        apply_edits,
    )
    from src.core.intelligence.skill_validator import validate_skill_candidate

    # Rate limiting: prevent rapid-fire edits to the same skill
    cooldown_key = (owner_id, skill_name.lower())
    now = datetime.now(timezone.utc)

    # TTL eviction: remove stale entries to prevent unbounded growth
    stale_keys = [
        k
        for k, ts in _edit_cooldowns.items()
        if (now - ts).total_seconds() > _COOLDOWN_TTL_SECONDS
    ]
    for k in stale_keys:
        del _edit_cooldowns[k]

    last_edit = _edit_cooldowns.get(cooldown_key)
    if last_edit is not None:
        elapsed = (now - last_edit).total_seconds()
        cooldown_sec = settings.skill_edit_cooldown_sec
        if elapsed < cooldown_sec:
            remaining = int(cooldown_sec - elapsed)
            return {
                "success": False,
                "error": f"Rate limited: wait {remaining}s before editing {skill_name!r} again",
                "cooldown_remaining": remaining,
            }
    _edit_cooldowns[cooldown_key] = now

    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        skill = await get_skill_by_name(session, owner, skill_name)
        if skill is None:
            return {"success": False, "error": f"Skill {skill_name!r} not found"}

        # Build the edit
        try:
            op = EditOp(edit_op)
        except ValueError:
            return {"success": False, "error": f"Invalid edit operation: {edit_op!r}"}

        edit = SkillEdit(
            op=op,
            target=edit_target,
            content=edit_content,
            reason=edit_reason,
        )

        # Apply bounded edits
        result = apply_edits(
            skill.body,
            [edit],
            edit_budget=settings.skill_edit_budget,
            current_version=skill.version or "1.0.0",
        )

        if not result.success:
            # Save to rejected-edits buffer
            rejected = skill.rejected_edits_json or []
            for rejected_edit, reason in result.rejected_edits:
                rejected.append(
                    {
                        **rejected_edit.to_dict(),
                        "reason": reason,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
            skill.rejected_edits_json = rejected[-10:]

            await session.flush()
            return {
                "success": False,
                "error": "Edit could not be applied",
                "rejected_edits": [
                    {"edit": e.to_dict(), "reason": r} for e, r in result.rejected_edits
                ],
            }

        # Validation gate (unless skipped for manual edits)
        validation_passed = True
        validation_summary = ""

        if not skip_validation and settings.skill_validation_enabled:
            validation = await validate_skill_candidate(
                owner_id,
                skill_name,
                result.new_body,
                is_edit=True,
                original_body=skill.body,
            )
            validation_passed = validation.accepted
            validation_summary = validation.summary

            if not validation_passed:
                # Save to rejected-edits buffer
                rejected = skill.rejected_edits_json or []
                rejected.append(
                    {
                        **edit.to_dict(),
                        "reason": f"Validation failed: {validation.reason}",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                skill.rejected_edits_json = rejected[-10:]
                await session.flush()

                return {
                    "success": False,
                    "error": "Validation gate rejected the edit",
                    "validation": validation_summary,
                    "rejected_edits": [
                        {"edit": edit.to_dict(), "reason": validation.reason}
                    ],
                }

            skill.validation_score = validation.score_after

            # Auto-rollback if score dropped below threshold
            if validation.score_after < 0.3 and skill.best_body:
                # Score is critically low — rollback to best_body
                skill.body = skill.best_body
                skill.validation_score = None  # Will be recalculated
                logger.warning(
                    "auto-rollback: score %.2f < 0.3 for %r, reverting to best_body",
                    validation.score_after,
                    skill_name,
                )
                # Still record the edit in history as "auto-rollback"
                history = skill.edit_history_json or []
                history.append(
                    {
                        "op": "auto-rollback",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "reason": f"Score dropped to {validation.score_after:.2f} (< 0.3 threshold)",
                    }
                )
                skill.edit_history_json = history[-20:]
                await session.flush()
                return {
                    "success": False,
                    "error": "Auto-rollback: score dropped below threshold",
                    "auto_rolled_back": True,
                    "validation": validation_summary,
                }

            # Only update best_body AFTER rollback check (to avoid overwriting
            # best_body with a low-score improvement that would then be rolled back)
            if validation.score_delta > 0 and validation.score_after >= 0.3:
                skill.best_body = result.new_body

        # Apply the edit
        old_version = skill.version or "1.0.0"
        from src.core.intelligence.skill_editor import bump_version

        new_version = bump_version(old_version, result.version_bump)

        # Update edit history
        history = skill.edit_history_json or []
        for applied_edit in result.applied_edits:
            history.append(
                {
                    **applied_edit.to_dict(),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "version_before": old_version,
                    "version_after": new_version,
                }
            )
        skill.edit_history_json = history[-20:]  # Keep last 20 edits

        # Apply changes
        skill.body = result.new_body
        skill.version = new_version
        skill.updated_at = datetime.now(timezone.utc)
        await session.flush()

        # Hot-reload: invalidate skill cache so next prompt uses updated skill
        from src.core.context_cache import invalidate as cache_invalidate

        await cache_invalidate(f"skills:{owner_id}:")

        return {
            "success": True,
            "new_version": new_version,
            "applied_edits": [e.to_dict() for e in result.applied_edits],
            "rejected_edits": [
                {"edit": e.to_dict(), "reason": r} for e, r in result.rejected_edits
            ],
            "validation": validation_summary,
        }


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


async def decay_stale_skills(session, telegram_id: int) -> int:
    """Авто-отключение навыков с упавшим success_rate.

    Критерии отключения:
    - enabled=True
    - usage_count >= 10 (достаточно данных)
    - success_rate < 0.3 (только 30% использований успешны)

    Возвращает количество отключённых навыков.
    """
    from src.db.models._learning import Skill

    result = await session.execute(
        select(Skill).where(
            Skill.user_id == telegram_id,
            Skill.enabled,
        )
    )
    skills = result.scalars().all()

    decayed = 0
    for skill in skills:
        usage_count = (skill.success_count or 0) + (skill.failure_count or 0)
        if usage_count < 10:
            continue

        success_rate = (skill.success_count or 0) / usage_count
        if success_rate < 0.3:
            skill.enabled = False
            old_desc = skill.description or ""
            skill.description = (
                old_desc + f" [DECAYED: success_rate={success_rate:.0%}]"
            )

            import logging

            logger = logging.getLogger(__name__)
            logger.info(
                f"Skill '{skill.name}' decayed: success_rate={success_rate:.0%} ({skill.success_count}/{usage_count})"
            )
            decayed += 1

    if decayed:
        await session.flush()

    return decayed


# ── background loop ──────────────────────────────────────────────────


async def curator_loop(owner_telegram_id: int) -> None:
    """Background loop: every 6 hours run auto-approval + suggestions + rollback.

    Runs:
        1. ``auto_approve_high_confidence()``
        2. ``suggest_skills_from_trajectories(owner_telegram_id)``
        3. ``propose_skills_from_analysis(owner_telegram_id)``
        4. ``rollback_all_regressed()`` — V3: auto-rollback regressed skills
    """
    from src.core.intelligence.skill_compact_optimizer import rollback_all_regressed
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

        # V3: Auto-rollback regressed skills (0 токенов)
        try:
            rolled_back = await rollback_all_regressed(owner_telegram_id)
            if rolled_back:
                await notification_queue.enqueue(
                    topic="skills",
                    category="curator",
                    priority=3,
                    text=(
                        f"♻️ Curator auto-rolled back {rolled_back} regressed skill(s). "
                        f"Check /skills for details."
                    ),
                )
        except Exception:
            logger.exception("curator_loop: rollback_all_regressed failed")

        # Health decay: отключаем мёртвые навыки
        try:
            async with get_session() as session:
                decayed = await decay_stale_skills(session, owner_telegram_id)
            if decayed:
                logger.info(
                    f"Decayed {decayed} stale skills for user {owner_telegram_id}"
                )
        except Exception:
            logger.exception("curator_loop: decay_stale_skills failed")

        await asyncio.sleep(interval_sec)


# ── task registration ────────────────────────────────────────────────

task_manager.register(
    "skill-curator",
    partial(curator_loop, settings.owner_telegram_id),
)
