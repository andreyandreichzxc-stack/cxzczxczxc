"""Prompt-level procedural skills for Asist.

V1 skills are not executable plugins. They are compact reusable procedures
injected into prompts when their triggers match the current request.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Iterable

from src.config import settings
from src.core.scheduling.notification_queue import notification_queue
from src.db.models import Skill, Trajectory
from src.db.repo import (
    add_skill_usage,
    get_or_create_user,
    list_skills,
    upsert_skill,
)
from src.db.session import get_session
from src.llm.base import TaskType

logger = logging.getLogger(__name__)


def _matches(text: str, patterns: Iterable[str] | None) -> int:
    if not patterns:
        return 0
    score = 0
    low = text.lower()
    for pattern in patterns:
        if not pattern:
            continue
        p = str(pattern).strip()
        if not p:
            continue
        try:
            if re.search(p, text, flags=re.IGNORECASE):
                score += 3
                continue
        except re.error:
            pass
        if p.lower() in low:
            score += 2
    return score


def _extract_json_from_response(text: str) -> str:
    m = re.search(r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    brace_m = re.search(r"\{[\s\S]*\}", text)
    if brace_m:
        return brace_m.group(0)
    return text.strip()


async def list_relevant_skills(
    telegram_id: int,
    user_text: str,
    route_mode: str | None = None,
    limit: int = 5,
) -> list[Skill]:
    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)
        skills = await list_skills(
            session,
            owner,
            enabled=True,
            review_status="approved",
            limit=100,
        )

    ranked: list[tuple[int, Skill]] = []
    for skill in skills:
        patterns = skill.trigger_patterns_json or []
        score = _matches(user_text, patterns)
        if route_mode and route_mode.lower() in [str(p).lower() for p in patterns]:
            score += 2
        if score:
            ranked.append((score + (skill.success_count or 0), skill))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [skill for _, skill in ranked[:limit]]


def format_skill_index(skills: list[Skill]) -> str:
    if not skills:
        return ""
    lines = ["<skill_index>"]
    for skill in skills[:5]:
        desc = (skill.description or "").strip()
        header = f"- {skill.name}"
        # V2: show version if not default
        ver = skill.version or "1.0.0"
        if ver != "1.0.0":
            header += f" v{ver}"
        if desc:
            header += f": {desc[:160]}"
        lines.append(header)
        body = (skill.body or "").strip()
        if body:
            lines.append(f"  procedure: {body[:700]}")
    lines.append("</skill_index>")

    # Inject rejected edits feedback for LLM to avoid repeating failed patterns
    from src.core.intelligence.skill_editor import format_rejected_edits

    for skill in skills[:5]:
        if skill.rejected_edits_json:
            feedback = format_rejected_edits(skill.rejected_edits_json)
            if feedback:
                lines.append("")
                lines.append(feedback)

    return "\n".join(lines)


async def build_skill_index(
    telegram_id: int,
    user_text: str,
    route_mode: str | None = None,
    limit: int = 5,
) -> tuple[str, list[dict]]:
    from src.core.context_cache import get as cache_get

    cached = await cache_get(f"skills:{telegram_id}:{route_mode}")
    if cached is not None:
        return cached

    skills = await list_relevant_skills(telegram_id, user_text, route_mode, limit)
    result = (
        format_skill_index(skills),
        [{"id": s.id, "name": s.name, "route_mode": route_mode} for s in skills],
    )

    from src.core.context_cache import put as cache_put

    await cache_put(f"skills:{telegram_id}:{route_mode}", result, ttl=30)
    return result


async def record_skill_usage(
    telegram_id: int,
    skill_id: int,
    trajectory_id: int | None,
    success: bool,
) -> None:
    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)
        skill = await session.get(Skill, skill_id)
        if skill is None or skill.user_id != owner.id:
            return
        await add_skill_usage(
            session,
            owner,
            skill,
            trajectory_id=trajectory_id,
            success=success,
        )


async def record_skill_usages(
    telegram_id: int,
    used_skills: list[dict] | None,
    trajectory_id: int | None,
    success: bool,
) -> None:
    if not used_skills:
        return
    for item in used_skills:
        skill_id = item.get("id") if isinstance(item, dict) else None
        if skill_id:
            await record_skill_usage(telegram_id, int(skill_id), trajectory_id, success)


def _safe_skill_name(route_mode: str, intent_name: str) -> str:
    base = f"{route_mode or 'general'}_{intent_name or 'chat'}"
    base = re.sub(r"[^a-zA-Z0-9_а-яА-Я-]+", "_", base).strip("_")
    return base[:96] or "general_chat"


async def suggest_skills_from_trajectories(telegram_id: int) -> int:
    """Create low-risk pending skills from repeated successful trajectories."""
    from sqlalchemy import select

    since = datetime.now(timezone.utc) - timedelta(days=1)
    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)
        rows = (
            (
                await session.execute(
                    select(Trajectory)
                    .where(
                        Trajectory.user_id == owner.id,
                        Trajectory.success,
                        Trajectory.created_at >= since,
                    )
                    .order_by(Trajectory.created_at.desc())
                    .limit(200)
                )
            )
            .scalars()
            .all()
        )

        buckets: Counter[tuple[str, str]] = Counter()
        examples: dict[tuple[str, str], Trajectory] = {}
        for row in rows:
            intent = row.intent_json or {}
            intent_name = str(intent.get("intent") or "chat")
            key = (row.route_mode or "unknown", intent_name)
            buckets[key] += 1
            examples.setdefault(key, row)

        created = 0
        # Load existing skill names once to avoid N+1 queries
        all_skills = await list_skills(session, owner, limit=200)
        existing_names = {s.name for s in all_skills}

        for (route_mode, intent_name), count in buckets.items():
            if count < 3:
                continue
            name = _safe_skill_name(route_mode, intent_name)
            if name in existing_names:
                continue
            sample = examples[(route_mode, intent_name)]
            body = (
                f"When route_mode={route_mode} and intent={intent_name}, prefer the "
                "shortest successful path used in recent trajectories. Preserve user "
                "intent, avoid inventing contacts, and ask a clarify question when "
                "required inputs are missing."
            )
            await upsert_skill(
                session,
                owner,
                name=name,
                description=f"Auto-suggested from {count} successful recent turns.",
                trigger_patterns_json=[
                    route_mode,
                    intent_name,
                    sample.request_text[:80],
                ],
                body=body,
                enabled=False,
                review_status="pending",
            )
            created += 1
        return created


async def _light_analysis(
    owner_id: int,
    messages: list[dict],
    hint_patterns: list[str] | None = None,
) -> list[dict]:
    """Дешёвый анализ (~10 сообщений, ~500 токенов).

    Ищет только явные, однозначные паттерны. Если ничего не находит —
    deep-анализ не запускается (экономия 3000+ токенов).

    Hint-паттерны (regex, 0 токенов) передаются в начало списка сообщений
    как контекст — существующий агент их обработает.
    """
    from src.agents.skill_creator_agent import propose as agent_propose
    from src.llm.router import build_provider

    try:
        async with get_session() as session:
            owner = await get_or_create_user(session, owner_id)
            provider = await build_provider(
                session, owner, purpose="background", task_type=TaskType.SKILLS
            )

        # Light: только последние 10 сообщений (в 5 раз меньше токенов)
        light_messages = messages[:10]

        # Prepend hint as synthetic message (exploits existing agent interface)
        if hint_patterns:
            hint_text = (
                "[PRE-ANALYSIS] Обнаружены повторяющиеся паттерны: "
                + "; ".join(hint_patterns)
            )
            light_messages.insert(
                0,
                {
                    "text": hint_text[:300],
                    "is_outgoing": True,
                    "timestamp": "",
                },
            )

        proposals = await agent_propose(provider, light_messages)

        if not proposals:
            logger.debug("_light_analysis: no patterns found")
            return []

        logger.info("_light_analysis: found %d candidate(s)", len(proposals))
        return proposals

    except Exception:
        logger.exception("_light_analysis: failed")
        return []


async def _deep_analysis(
    owner_id: int,
    messages: list[dict],
    hint_patterns: list[str],
) -> list[dict]:
    """Полный анализ (~50 сообщений, ~3000 токенов).

    Запускается ТОЛЬКО если light-анализ что-то нашёл.
    """
    from src.agents.skill_creator_agent import propose as agent_propose
    from src.llm.router import build_provider

    try:
        async with get_session() as session:
            owner = await get_or_create_user(session, owner_id)
            provider = await build_provider(
                session, owner, purpose="background", task_type=TaskType.SKILLS
            )

        # Deep: все 50 сообщений
        deep_messages = messages[:50]

        # Prepend hint with light analysis results as context
        hint_text = (
            "[DEEP-ANALYSIS] Light-анализ обнаружил паттерны: "
            + "; ".join(hint_patterns[:10])
            + ". Твоя задача: глубокий анализ. Предложи bounded edits или новые навыки."
        )
        deep_messages.insert(
            0,
            {
                "text": hint_text[:300],
                "is_outgoing": True,
                "timestamp": "",
            },
        )

        proposals = await agent_propose(provider, deep_messages)

        if not proposals:
            logger.debug("_deep_analysis: no proposals after deep analysis")
            return []

        logger.info("_deep_analysis: proposed %d changes", len(proposals))
        return proposals

    except Exception:
        logger.exception("_deep_analysis: failed")
        return []


async def propose_skills_from_analysis(
    owner_id: int,
    *,
    tier: str = "auto",
    force: bool = False,
) -> list[dict]:
    """Tiered skill analysis: gatekeeper → light → deep.

    V3: Адаптировано под real-time агента с минимизацией токенов.

    Modes:
    - auto (default): gatekeeper решает, запускать ли. Если да:
        extract_light_patterns → light_analysis → deep_analysis (только если light нашёл)
    - light: только дешёвый анализ (10 сообщений, medium-модель)
    - deep: полный анализ (50 сообщений, heavy-модель). Пропускает gatekeeper.

    Returns:
        Список созданных навыков: [{"name": str, "id": int, "confidence": float}, ...]
    """
    from src.agents.skill_creator_agent import propose as agent_propose
    from src.config import settings as cfg
    from src.core.intelligence.skill_gatekeeper import (
        detect_skill_conflicts,
        extract_light_patterns,
        format_conflict_warning,
        get_gatekeeper,
    )
    from src.core.intelligence.skill_validator import validate_skill_candidate
    from src.db.repo import fetch_my_messages_global
    from src.llm.router import build_provider

    # ── Tier routing ──
    if tier == "auto" and not force:
        gatekeeper = get_gatekeeper()
        should_run, gate_reason = await gatekeeper.should_analyze(owner_id)
        if not should_run:
            logger.debug(
                "propose_skills_from_analysis: gatekeeper skip — %s", gate_reason
            )
            return []
        logger.info(
            "propose_skills_from_analysis: gatekeeper approved — %s", gate_reason
        )

    # ── Fetch messages ──
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        messages_raw = await fetch_my_messages_global(session, owner, limit=50)
        recent_messages = [
            {
                "text": msg.text or "",
                "is_outgoing": msg.is_outgoing if hasattr(msg, "is_outgoing") else True,
                "timestamp": str(msg.date) if hasattr(msg, "date") else "",
            }
            for msg in messages_raw
        ]

    if not recent_messages:
        logger.debug("propose_skills_from_analysis: no messages to analyze")
        return []

    proposals: list[dict] = []

    # ── Tiered analysis ──
    if tier == "light":
        proposals = await _light_analysis(owner_id, recent_messages)

    elif tier == "deep":
        # Deep без light-префильтра (форсированный режим)
        async with get_session() as session:
            owner = await get_or_create_user(session, owner_id)
            provider = await build_provider(
                session, owner, purpose="background", task_type=TaskType.SKILLS
            )
        proposals = await agent_propose(provider, recent_messages[:50])

    else:  # auto
        # Step 1: Free pre-analysis — extract patterns
        msg_texts = [m["text"] for m in recent_messages]
        hint_patterns = extract_light_patterns(msg_texts)

        if not hint_patterns:
            logger.debug(
                "propose_skills_from_analysis: no regex patterns found, skipping"
            )
            return []

        # Step 2: Light analysis (~500 tokens)
        proposals = await _light_analysis(owner_id, recent_messages, hint_patterns)

        # Step 3: Deep analysis only if light found something (~3000 tokens)
        if proposals:
            deep_proposals = await _deep_analysis(
                owner_id, recent_messages, hint_patterns
            )
            if deep_proposals:
                proposals = deep_proposals  # Replace with richer deep results

    if not proposals:
        return []

    # ── Filter & create skills ──
    created_skills: list[dict] = []

    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        all_skills = await list_skills(session, owner, limit=200)
        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue
            confidence = proposal.get("confidence", 0)
            if not isinstance(confidence, (int, float)) or confidence <= 0.7:
                continue

            name = str(proposal.get("name", "")).strip()
            if not name:
                continue

            # Check for duplicates
            existing = [s for s in all_skills if s.name.lower() == name.lower()]
            if existing:
                logger.debug(
                    "propose_skills_from_analysis: skill %r already exists", name
                )
                continue

            # Проверка конфликтов trigger-паттернов с существующими навыками
            triggers = proposal.get("trigger_patterns") or []
            if triggers:
                skill_tuples = [(s.name, s.trigger_patterns_json) for s in all_skills]
                conflicts = detect_skill_conflicts(name, triggers, skill_tuples)
                if conflicts:
                    conflict_msg = format_conflict_warning(conflicts)
                    logger.info("Skill '%s' trigger conflicts: %s", name, conflict_msg)

            body = str(proposal.get("body", ""))

            # Validation gate
            validation_result = None
            if cfg.skill_validation_enabled:
                validation_result = await validate_skill_candidate(
                    owner_id,
                    name,
                    body,
                )
                if not validation_result.accepted:
                    try:
                        await upsert_skill(
                            session,
                            owner,
                            name=name[:128],
                            description=str(proposal.get("description", "")),
                            trigger_patterns_json=proposal.get("trigger_patterns")
                            or [],
                            body=body,
                            enabled=False,
                            review_status="pending",
                        )
                        logger.info(
                            "propose_skills_from_analysis: created %r as pending "
                            "(validation: %s)",
                            name,
                            validation_result.summary,
                        )
                    except Exception:
                        logger.exception(
                            "propose_skills_from_analysis: failed to upsert skill %r",
                            name,
                        )
                    continue

            try:
                skill = await upsert_skill(
                    session,
                    owner,
                    name=name[:128],
                    description=str(proposal.get("description", "")),
                    trigger_patterns_json=proposal.get("trigger_patterns") or [],
                    body=body,
                    enabled=True,
                    review_status="approved",
                )
                if validation_result is not None:
                    skill.validation_score = validation_result.score_after
                    skill.best_body = body

                created_skills.append(
                    {
                        "name": name,
                        "id": skill.id,
                        "confidence": confidence,
                    }
                )
                logger.info(
                    "propose_skills_from_analysis: created skill %r (confidence=%.2f)",
                    name,
                    confidence,
                )
            except Exception:
                logger.exception(
                    "propose_skills_from_analysis: failed to upsert skill %r", name
                )

    return created_skills


async def propose_edits_from_corrections(
    telegram_id: int,
    provider=None,
    max_corrections: int = 5,
) -> list[dict]:
    """Анализирует недавние коррекции пользователя и предлагает правки в навыки.

    Если пользователь 3+ раза исправил поведение, связанное с навыком X —
    предложить edit в навык X.

    Возвращает список proposed edits (для последующей валидации).
    Только если нашли связь correction → skill.
    """
    import json as _json

    from src.core.intelligence.correction_learner import get_recent_corrections
    from src.llm.base import ChatMessage
    from src.llm.router import build_provider

    corrections = await get_recent_corrections(telegram_id, limit=max_corrections)
    if not corrections:
        return []

    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)
        skills = await list_skills(session, owner, enabled=True, limit=100)

    if not skills:
        return []

    # ── Map corrections to skills ──
    skill_corrections: dict[int, list[dict]] = {}

    for corr in corrections:
        orig = (corr.get("original") or "").lower()
        corr_text = (corr.get("corrected") or "").lower()
        combined = f"{orig} {corr_text}"

        best_skill = None
        best_score = 0

        for skill in skills:
            score = 0
            skill_name_norm = skill.name.lower().replace("_", " ").replace("-", " ")
            if skill_name_norm in combined:
                score += 5
            patterns = skill.trigger_patterns_json or []
            if patterns:
                score += _matches(combined, patterns)
            desc = (skill.description or "").lower()
            if desc and any(w in combined for w in desc.split() if len(w) > 4):
                score += 1
            if score > best_score:
                best_score = score
                best_skill = skill

        if best_skill and best_score >= 3:
            sid = best_skill.id
            skill_corrections.setdefault(sid, []).append(corr)

    # ── Only skills with 3+ corrections ──
    eligible = {sid: cs for sid, cs in skill_corrections.items() if len(cs) >= 3}
    if not eligible:
        return []

    # ── Build provider if needed ──
    if provider is None:
        async with get_session() as session:
            owner = await get_or_create_user(session, telegram_id)
            provider = await build_provider(
                session, owner, purpose="background", task_type=TaskType.SKILLS
            )
    if provider is None:
        return []

    # ── LLM: propose edits (lightweight, ~300 tokens each) ──
    proposals: list[dict] = []
    for sid, corrs in list(eligible.items())[:3]:
        skill = next((s for s in skills if s.id == sid), None)
        if not skill:
            continue

        correction_lines = []
        for i, c in enumerate(corrs[-5:], 1):
            correction_lines.append(
                f"{i}. Bot: {c.get('original', '')[:200]}\n"
                f"   Corrected to: {c.get('corrected', '')[:200]}"
            )

        prompt = (
            "User provided corrections that affect this skill. "
            "Propose ONE minimal edit to align the skill with user preferences.\n\n"
            f"SKILL: {skill.name}\n"
            f"DESC:  {skill.description or 'N/A'}\n"
            f"BODY:  {(skill.body or '')[:400]}\n\n"
            "CORRECTIONS:\n" + "\n".join(correction_lines) + "\n\n"
            "EDIT TYPES:\n"
            "- append: add to end of body\n"
            "- insert_after: insert after a line (provide target line text)\n"
            "- replace: replace old text with new\n"
            "- delete: remove text\n\n"
            "Return ONLY JSON (no markdown):\n"
            '{"op":"...", "target":"...", "content":"...", "reason":"..."}'
        )

        try:
            response = await provider.chat(
                [ChatMessage(role="user", content=prompt)], task_type=TaskType.SKILLS
            )
            json_str = _extract_json_from_response(response)
            proposal = _json.loads(json_str)
            proposal["skill_id"] = skill.id
            proposal["skill_name"] = skill.name
            proposals.append(proposal)
            logger.info(
                "propose_edits_from_corrections: proposed edit for %r (%d corrections)",
                skill.name,
                len(corrs),
            )
        except Exception:
            logger.debug(
                "propose_edits_from_corrections: LLM failed for %r",
                skill.name,
                exc_info=True,
            )

    return proposals


async def skill_optimizer_loop(telegram_id: int) -> None:
    """Фоновый цикл оптимизации навыков.

    V3: Gatekeeper-controlled. Вместо time-based cooldown использует
    message delta + trajectory count для принятия решения. Экономия: -93% токенов.
    """
    from src.core.intelligence.skill_gatekeeper import get_gatekeeper

    _gatekeeper = get_gatekeeper()

    while True:
        # Step 1: Trajectory-based suggestions (бесплатно, всегда)
        try:
            created = await suggest_skills_from_trajectories(telegram_id)
            if created:
                await notification_queue.enqueue(
                    topic="skills",
                    category="self_evolution",
                    priority=2,
                    text=f"Found {created} new skill suggestions. Open /evolve.",
                )
        except Exception:
            logger.exception("skill_optimizer_loop trajectory analysis failed")

        # Step 2: LLM-based proposals — только если gatekeeper разрешает
        try:
            should_run, gate_reason = await _gatekeeper.should_analyze(telegram_id)
            if should_run:
                proposed = await propose_skills_from_analysis(
                    telegram_id, tier="auto", force=True
                )
                if proposed:
                    names = [s["name"] for s in proposed]
                    await notification_queue.enqueue(
                        topic="skills",
                        category="self_evolution",
                        priority=3,
                        text=(
                            f"Skill Creator создал {len(proposed)} навыков: "
                            f"{', '.join(names[:5])}. "
                            f"Уже активны в /skills."
                        ),
                    )
            else:
                logger.debug("skill_optimizer_loop: gatekeeper skip — %s", gate_reason)
        except Exception:
            logger.exception("skill_optimizer_loop skill creator analysis failed")

        # Step 3: Correction-based edits — analyse recent user corrections
        try:
            edits = await propose_edits_from_corrections(
                telegram_id,
                max_corrections=5,
            )
            if edits:
                logger.info(
                    "skill_optimizer_loop: proposed %d edits from corrections",
                    len(edits),
                )
            else:
                logger.debug("skill_optimizer_loop: no edits from corrections")
        except Exception:
            logger.exception("skill_optimizer_loop correction-based analysis failed")

        await asyncio.sleep(settings.skill_optimizer_interval_sec)


from functools import partial
from src.core.infra.task_manager import task_manager

task_manager.register(
    "skill-optimizer", partial(skill_optimizer_loop, settings.owner_telegram_id)
)
