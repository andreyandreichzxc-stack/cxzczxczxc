"""Skill repository — Skill, SkillUsage, Trajectory."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import (
    Skill,
    SkillUsage,
    Trajectory,
)

logger = logging.getLogger(__name__)


async def add_trajectory(
    session: AsyncSession,
    user,
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
) -> Trajectory:
    row = Trajectory(
        user_id=user.id,
        request_text=request_text[:8000],
        route_mode=route_mode,
        intent_json=intent_json,
        actions_json=actions_json,
        used_skills_json=used_skills_json,
        memory_ids_json=memory_ids_json,
        response_text=response_text[:8000] if response_text else None,
        success=success,
        error=error[:4000] if error else None,
        latency_ms=latency_ms,
    )
    session.add(row)
    await session.flush()
    return row


async def list_trajectories(
    session: AsyncSession,
    user,
    *,
    only_errors: bool = False,
    limit: int = 20,
) -> list[Trajectory]:
    q = select(Trajectory).where(Trajectory.user_id == user.id)
    if only_errors:
        q = q.where(Trajectory.success.is_(False))
    q = q.order_by(Trajectory.created_at.desc()).limit(limit)
    r = await session.execute(q)
    return list(r.scalars().all())


async def upsert_skill(
    session: AsyncSession,
    user,
    *,
    name: str,
    description: str | None = None,
    trigger_patterns_json: list | None = None,
    body: str,
    enabled: bool = True,
    review_status: str = "approved",
) -> Skill:
    # Feature 3: YAML frontmatter parsing
    # Если description содержит YAML frontmatter (---...---),
    # парсим метаданные и сохраняем в trigger_patterns_json как __yaml__
    clean_description = description
    yaml_metadata: dict[str, object] = {}
    if description and description.strip().startswith("---"):
        try:
            from src.core.intelligence.skill_yaml import extract_frontmatter_metadata

            yaml_metadata, clean_description = extract_frontmatter_metadata(description)
        except Exception:
            logger.debug("upsert_skill: YAML frontmatter parse skipped", exc_info=True)
            clean_description = description

    # Собираем trigger_patterns_json: базовые паттерны + YAML метаданные
    patterns = list(trigger_patterns_json or [])
    if yaml_metadata:
        # Добавляем теги из YAML как паттерны
        tags = yaml_metadata.get("tags", [])
        if isinstance(tags, list):
            for tag in tags:
                if isinstance(tag, str) and tag.strip() not in patterns:
                    patterns.append(tag.strip())
        # Сохраняем структурированные метаданные как __yaml__
        # Убираем дубликат __yaml__ если уже есть
        patterns = [
            p for p in patterns if not (isinstance(p, dict) and "__yaml__" in p)
        ]
        patterns.append({"__yaml__": yaml_metadata})

    result = await session.execute(
        select(Skill).where(
            Skill.user_id == user.id,
            func.lower(Skill.name) == name.lower().strip(),
        )
    )
    skill = result.scalar_one_or_none()
    if skill is None:
        skill = Skill(
            user_id=user.id,
            name=name.strip(),
            description=clean_description,
            trigger_patterns_json=patterns,
            body=body,
            enabled=enabled,
            review_status=review_status,
        )
        session.add(skill)
    else:
        skill.description = clean_description
        skill.trigger_patterns_json = patterns
        skill.body = body
        skill.enabled = enabled
        skill.review_status = review_status
        skill.updated_at = datetime.now(timezone.utc)
    await session.flush()
    try:
        from src.core.infra.hooks import hooks

        await hooks.emit(
            "on_skill_created", skill_name=skill.name, user_id=user.telegram_id
        )
    except Exception:
        pass  # hooks are optional, never break core flow
    # Invalidate skill index cache so next prompt picks up changes
    try:
        from src.core.context_cache import invalidate as cache_invalidate

        await cache_invalidate(f"skills:{user.telegram_id}:")
    except Exception:
        pass
    return skill


async def list_skills(
    session: AsyncSession,
    user,
    *,
    enabled: bool | None = None,
    review_status: str | None = None,
    limit: int = 50,
) -> list[Skill]:
    q = select(Skill).where(Skill.user_id == user.id)
    if enabled is not None:
        q = q.where(Skill.enabled == enabled)
    if review_status:
        q = q.where(Skill.review_status == review_status)
    q = q.order_by(Skill.success_count.desc(), Skill.updated_at.desc()).limit(limit)
    r = await session.execute(q)
    return list(r.scalars().all())


async def get_skill_by_name(session: AsyncSession, user, name: str) -> Skill | None:
    r = await session.execute(
        select(Skill).where(
            Skill.user_id == user.id,
            func.lower(Skill.name) == name.lower().strip(),
        )
    )
    return r.scalar_one_or_none()


async def set_skill_enabled(
    session: AsyncSession,
    user,
    name: str,
    enabled: bool,
    *,
    review_status: str | None = None,
) -> Skill | None:
    skill = await get_skill_by_name(session, user, name)
    if skill is None:
        return None
    skill.enabled = enabled
    if review_status is not None:
        skill.review_status = review_status
    skill.updated_at = datetime.now(timezone.utc)
    await session.flush()
    # Invalidate skill index cache so next prompt picks up changes
    try:
        from src.core.context_cache import invalidate as cache_invalidate

        await cache_invalidate(f"skills:{user.telegram_id}:")
    except Exception:
        pass
    return skill


async def add_skill_usage(
    session: AsyncSession,
    user,
    skill,
    *,
    trajectory_id: int | None = None,
    success: bool = True,
) -> SkillUsage:
    usage = SkillUsage(
        user_id=user.id,
        skill_id=skill.id,
        trajectory_id=trajectory_id,
        success=success,
    )
    session.add(usage)
    if success:
        skill.success_count = (skill.success_count or 0) + 1
    else:
        skill.failure_count = (skill.failure_count or 0) + 1
    skill.last_used_at = datetime.now(timezone.utc)
    skill.updated_at = datetime.now(timezone.utc)
    await session.flush()
    return usage
