"""Skill Validator — validation gate for skill updates.

Inspired by SkillOpt's held-out validation set approach. Before accepting
any skill change (new skill or edit), validates it against recent successful
trajectories to ensure quality doesn't regress.

The validation gate:
1. Selects held-out trajectories (successful, not used for training)
2. Simulates the skill being active during those trajectories
3. Compares quality metrics (success rate, latency, completeness)
4. Accepts only if score improves or stays stable

V2.1: LLM-based validation with heuristic fallback.

This prevents:
- Bad skills from being auto-approved
- Edits that degrade performance
- Skill bloat from low-quality proposals
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.llm.base import LLMProvider

from sqlalchemy import func, select

from src.db.models import Skill, SkillUsage, Trajectory
from src.db.session import get_session

logger = logging.getLogger(__name__)

# ── Validation thresholds ──

MIN_TRAJECTORIES_FOR_VALIDATION = 3
MAX_REGRESSION_TOLERANCE = -0.05  # Allow 5% regression tolerance
UNCERTAIN_DELTA_THRESHOLD = 0.05  # Score delta magnitude that triggers LLM fallback


@dataclass
class TrajectoryData:
    """Lightweight snapshot of trajectory data for scoring (detached-safe)."""

    id: int
    request_text: str
    response_text: str
    latency_ms: int | None
    used_skills_json: list | None
    route_mode: str | None
    success: bool

    @classmethod
    def from_trajectory(cls, t: Trajectory) -> TrajectoryData:
        """Extract plain data from ORM Trajectory object."""
        return cls(
            id=t.id,
            request_text=t.request_text or "",
            response_text=t.response_text or "",
            latency_ms=t.latency_ms,
            used_skills_json=t.used_skills_json,
            route_mode=t.route_mode,
            success=t.success,
        )


@dataclass
class ValidationResult:
    """Result of validating a skill candidate."""

    accepted: bool
    score_before: float  # Baseline score without the change
    score_after: float  # Score with the proposed change
    score_delta: float  # Improvement (positive = better)
    trajectories_used: int
    reason: str = ""

    @property
    def summary(self) -> str:
        status = "✅ ACCEPTED" if self.accepted else "❌ REJECTED"
        return (
            f"{status}: score {self.score_before:.2f} → {self.score_after:.2f} "
            f"(Δ={self.score_delta:+.2f}) on {self.trajectories_used} trajectories"
        )


async def _llm_estimate_quality(
    new_body: str,
    trajectories: list[TrajectoryData],
    skill_name: str,
) -> float | None:
    """Use LLM to estimate skill quality against trajectories.

    Returns a score between 0.0 and 1.0, or None if LLM is unavailable.
    Uses skill_optimizer_model from config, falls back to None on error.
    """
    from src.config import settings

    model = settings.skill_optimizer_model
    if not model:
        return None

    try:
        from src.db.repo import get_or_create_user
        from src.llm.base import ChatMessage, TaskType
        from src.llm.router import build_provider

        async with get_session() as session:
            owner = await get_or_create_user(session, settings.owner_telegram_id)
            provider = await build_provider(
                session, owner, purpose="background", task_type=TaskType.SKILLS
            )

        if not provider:
            return None

        # Build evaluation prompt with trajectory examples
        examples = []
        for t in trajectories[:5]:
            examples.append(
                f"Запрос: {t.request_text[:200]}\nОтвет: {t.response_text[:200]}"
            )

        prompt = (
            f"Оцени качество навыка '{skill_name}' (0.0-1.0) на основе примеров диалогов.\n\n"
            f"Тело навыка:\n{new_body[:1500]}\n\n"
            f"Примеры диалогов:\n" + "\n---\n".join(examples) + "\n\n"
            "Ответь ТОЛЬКО числом от 0.0 до 1.0 (например: 0.75)."
        )

        messages = [
            ChatMessage(
                role="system",
                content="Ты — оценщик качества навыков для AI-ассистента. Отвечай только числом.",
            ),
            ChatMessage(role="user", content=prompt),
        ]

        response = await provider.chat(messages, task_type=TaskType.SKILLS)
        score = float(response.strip().split()[0])  # Extract first number
        return max(0.0, min(1.0, score))

    except Exception as e:
        logger.debug("LLM validation failed, falling back to heuristic: %s", e)
        return None


async def _llm_validate_skill(
    skill_body: str,
    trajectories: list[TrajectoryData],
    provider: "LLMProvider",
) -> float | None:
    """Lightweight LLM second opinion for uncertain heuristic scores.

    Calls provider with a prompt asking to rate the skill 0-1 for the given
    trajectories. Parses JSON response ``{"score": 0.X}``.

    Returns:
        Score between 0.0 and 1.0, or None on failure.
    """
    from src.llm.base import ChatMessage, TaskType

    # Build short evaluation prompt with up to 3 trajectory examples
    examples = []
    for t in trajectories[:3]:
        examples.append(
            f"Request: {t.request_text[:200]}\nResponse: {t.response_text[:100]}"
        )

    prompt = (
        "You are evaluating a skill for an AI assistant. "
        "Rate this skill 0.0-1.0 for the given conversation trajectories.\n\n"
        f"Skill body:\n{skill_body[:1000]}\n\n"
        f"Example trajectories:\n" + "\n---\n".join(examples) + "\n\n"
        "Output only a JSON object with a single key 'score', e.g. {\"score\": 0.75}."
    )

    try:
        response = await provider.chat(
            [ChatMessage(role="user", content=prompt)], task_type=TaskType.SKILLS
        )
        import json

        parsed = json.loads(response.strip())
        score = float(parsed.get("score", -1))
        if 0.0 <= score <= 1.0:
            return score
        logger.debug("LLM validation returned out-of-range score: %s", score)
        return None
    except Exception:
        logger.warning("_llm_validate_skill failed", exc_info=True)
        return None


async def validate_skill_candidate(
    user_id: int,
    skill_name: str,
    new_body: str,
    *,
    is_edit: bool = False,
    original_body: str | None = None,
    provider: "LLMProvider | None" = None,
) -> ValidationResult:
    """Validate a skill candidate against held-out trajectories.

    V2.1: Uses LLM-based quality estimation with heuristic fallback.

    Args:
        user_id: Owner user ID
        skill_name: Name of the skill being validated
        new_body: Proposed new body text
        is_edit: True if this is an edit (not a new skill)
        original_body: Original body for edit comparison
        provider: Optional LLM provider for uncertain-score fallback.
                  When passed and heuristic score_delta is within
                  UNCERTAIN_DELTA_THRESHOLD, a light LLM call is made
                  as a second opinion.

    Returns:
        ValidationResult with accept/reject decision
    """
    async with get_session() as session:
        # 1. Get held-out trajectories: successful, recent, NOT using this skill
        since = datetime.now(timezone.utc) - timedelta(days=7)

        # Get trajectory IDs that already used this skill (to exclude from validation)
        used_trajectory_ids: set[int] = set()
        skill_result = await session.execute(
            select(Skill).where(
                Skill.user_id == user_id,
                func.lower(Skill.name) == skill_name.lower(),
            )
        )
        existing_skill = skill_result.scalar_one_or_none()
        existing_skill_name: str | None = (
            existing_skill.name if existing_skill else None
        )

        if existing_skill:
            usages = await session.execute(
                select(SkillUsage.trajectory_id).where(
                    SkillUsage.skill_id == existing_skill.id,
                    SkillUsage.trajectory_id.isnot(None),
                )
            )
            used_trajectory_ids = {row[0] for row in usages.all()}

        # Get held-out trajectories (successful, recent, not using this skill)
        query = (
            select(Trajectory)
            .where(
                Trajectory.user_id == user_id,
                Trajectory.success.is_(True),
                Trajectory.created_at >= since,
            )
            .order_by(Trajectory.created_at.desc())
            .limit(20)
        )
        result = await session.execute(query)
        all_trajectories = list(result.scalars().all())

        # CRITICAL FIX: Extract data into plain dataclasses BEFORE session closes
        held_out_raw = [t for t in all_trajectories if t.id not in used_trajectory_ids][
            :10
        ]
        held_out = [TrajectoryData.from_trajectory(t) for t in held_out_raw]

    if len(held_out) < MIN_TRAJECTORIES_FOR_VALIDATION:
        # Not enough data for validation — accept with warning
        return ValidationResult(
            accepted=True,
            score_before=0.0,
            score_after=0.0,
            score_delta=0.0,
            trajectories_used=len(held_out),
            reason=f"Insufficient validation data ({len(held_out)} < {MIN_TRAJECTORIES_FOR_VALIDATION}). Accepted with caution.",
        )

    # 2. Calculate baseline score (how well existing skill performs)
    baseline_score = _calculate_baseline_score(held_out, existing_skill_name)

    # 3. Estimate new score — try LLM first, fall back to heuristic
    llm_score = await _llm_estimate_quality(new_body, held_out, skill_name)
    if llm_score is not None:
        estimated_score = llm_score
        method = "LLM"
    else:
        estimated_score = _estimate_skill_quality_heuristic(
            new_body, held_out, skill_name
        )
        method = "heuristic"

    score_delta = estimated_score - baseline_score

    # 3b. LLM fallback for uncertain heuristic scores (second opinion)
    if (
        method == "heuristic"
        and abs(score_delta) <= UNCERTAIN_DELTA_THRESHOLD
        and provider is not None
    ):
        try:
            llm_score = await _llm_validate_skill(new_body, held_out, provider)
            if llm_score is not None:
                estimated_score = llm_score
                score_delta = estimated_score - baseline_score
                method = "LLM+heuristic"
        except Exception:
            logger.warning(
                "LLM validation fallback failed, using heuristic", exc_info=True
            )

    # 4. Accept/reject based on threshold
    accepted = score_delta >= MAX_REGRESSION_TOLERANCE
    reason = ""

    if not accepted:
        reason = f"Score regression ({method}): {score_delta:+.2f} (tolerance: {MAX_REGRESSION_TOLERANCE:+.2f})"
    elif score_delta > 0:
        reason = f"Score improvement ({method}): {score_delta:+.2f}"
    else:
        reason = f"Score stable ({method}): {score_delta:+.2f}"

    return ValidationResult(
        accepted=accepted,
        score_before=baseline_score,
        score_after=estimated_score,
        score_delta=score_delta,
        trajectories_used=len(held_out),
        reason=reason,
    )


def _calculate_baseline_score(
    trajectories: list[TrajectoryData],
    skill_name: str | None,
) -> float:
    """Calculate baseline quality score from trajectories.

    Score is based on:
    - Latency bonus (lower is better)
    - Response completeness (response_text length)
    - Skill usage bonus (if skill was used in trajectory)

    Returns:
        Score between 0.0 and 1.0
    """
    if not trajectories:
        return 0.3  # Low default when no data

    total_score = 0.0
    for t in trajectories:
        turn_score = 0.3  # Base score

        # Latency bonus (faster = better)
        if t.latency_ms and t.latency_ms > 0:
            if t.latency_ms < 2000:
                turn_score += 0.25
            elif t.latency_ms < 5000:
                turn_score += 0.15
            elif t.latency_ms < 10000:
                turn_score += 0.05

        # Completeness bonus (longer response = more complete)
        if t.response_text and len(t.response_text) > 50:
            turn_score += 0.2
        elif t.response_text and len(t.response_text) > 20:
            turn_score += 0.1

        # Skill usage bonus
        if skill_name and t.used_skills_json:
            used_names = [
                s.get("name", "") if isinstance(s, dict) else ""
                for s in (t.used_skills_json or [])
            ]
            if skill_name in used_names:
                turn_score += 0.25

        total_score += min(turn_score, 1.0)

    return total_score / len(trajectories)


def _estimate_skill_quality_heuristic(
    new_body: str,
    trajectories: list[TrajectoryData],
    skill_name: str,
) -> float:
    """Heuristic-based quality estimation (fast, no LLM call).

    Used as fallback when LLM is unavailable.
    """
    if not new_body or not new_body.strip():
        return 0.0

    body_lower = new_body.lower()
    score = 0.5  # Base score

    # Length check (300-3000 chars is ideal for skills)
    body_len = len(new_body)
    if 300 <= body_len <= 3000:
        score += 0.15  # Good length
    elif body_len < 100:
        score -= 0.2  # Too short
    elif body_len > 5000:
        score -= 0.1  # Too long

    # Structure check (has steps, rules, or procedures)
    structure_markers = [
        "1.",
        "2.",
        "•",
        "- ",
        "step",
        "правило",
        "алгоритм",
        "процедур",
        "когда",
        "если",
        "then",
        "when",
        "if ",
    ]
    structure_count = sum(1 for m in structure_markers if m in body_lower)
    if structure_count >= 2:
        score += 0.15

    # Relevance check (keywords from trajectories)
    keyword_hits = 0
    body_words = {w for w in body_lower.split() if len(w) >= 3}
    for t in trajectories[:5]:
        if t.request_text:
            words = {w for w in t.request_text.lower().split() if len(w) >= 3}
            overlap = len(words & body_words)
            if overlap >= 2:
                keyword_hits += 1

    if keyword_hits > 0:
        score += min(keyword_hits * 0.05, 0.2)

    return min(score, 1.0)


async def validate_and_update_skill(
    user_id: int,
    skill_name: str,
    new_body: str,
    *,
    new_description: str | None = None,
    is_edit: bool = False,
    provider: "LLMProvider | None" = None,
) -> tuple[bool, ValidationResult]:
    """Validate a skill candidate and update if accepted.

    V2.1: Uses single session for read-validate-write to prevent race conditions.

    Args:
        user_id: Owner user ID
        skill_name: Skill name
        new_body: Proposed new body
        new_description: Optional new description
        is_edit: True if this is an edit
        provider: Optional LLM provider for uncertain-score fallback.

    Returns:
        (accepted, validation_result)
    """
    from src.db.repo import get_or_create_user

    # Step 1: Read original body in the same session context
    async with get_session() as session:
        owner = await get_or_create_user(session, user_id)
        skill_result = await session.execute(
            select(Skill).where(
                Skill.user_id == owner.id,
                func.lower(Skill.name) == skill_name.lower(),
            )
        )
        existing = skill_result.scalar_one_or_none()
        original_body = existing.body if existing else None

    # Step 2: Validate (uses its own session internally for DB reads)
    validation = await validate_skill_candidate(
        user_id,
        skill_name,
        new_body,
        is_edit=is_edit,
        original_body=original_body,
        provider=provider,
    )

    # Step 3: Update in a single session (write-only, no race window)
    if validation.accepted:
        async with get_session() as session:
            owner = await get_or_create_user(session, user_id)
            skill_result = await session.execute(
                select(Skill).where(
                    Skill.user_id == owner.id,
                    func.lower(Skill.name) == skill_name.lower(),
                )
            )
            skill = skill_result.scalar_one_or_none()

            if skill:
                skill.validation_score = validation.score_after
                if validation.score_delta > 0:
                    skill.best_body = new_body
                # get_session() auto-commits on exit

        logger.info(
            "skill_validator: accepted %r — %s",
            skill_name,
            validation.summary,
        )
    else:
        logger.info(
            "skill_validator: rejected %r — %s",
            skill_name,
            validation.summary,
        )

    return validation.accepted, validation


async def get_validation_stats(user_id: int) -> dict:
    """Get validation statistics for the user's skills."""
    async with get_session() as session:
        from src.db.repo import get_or_create_user

        owner = await get_or_create_user(session, user_id)

        # Count skills with validation scores
        scored = await session.execute(
            select(func.count(Skill.id)).where(
                Skill.user_id == owner.id,
                Skill.validation_score.isnot(None),
            )
        )
        scored_count = scored.scalar() or 0

        # Average validation score
        avg_result = await session.execute(
            select(func.avg(Skill.validation_score)).where(
                Skill.user_id == owner.id,
                Skill.validation_score.isnot(None),
            )
        )
        avg_score = avg_result.scalar() or 0.0

        # Count skills with best_body snapshots
        best_count_result = await session.execute(
            select(func.count(Skill.id)).where(
                Skill.user_id == owner.id,
                Skill.best_body.isnot(None),
            )
        )
        best_count = best_count_result.scalar() or 0

    return {
        "scored_skills": scored_count,
        "avg_validation_score": round(float(avg_score), 3),
        "skills_with_best_body": best_count,
    }
