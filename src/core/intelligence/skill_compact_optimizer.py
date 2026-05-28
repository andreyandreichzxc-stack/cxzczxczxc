"""Skill Compact Optimizer — три улучшения в одном файле.  0 токенов.

1. Auto-Rollback — откат к best_body при падении success_rate.
2. LR Scheduling — edit_budget зависит от зрелости навыка.
3. Edit Priority — FAILURE_FIX побеждает SUCCESS_BOOST.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import select

from src.db.session import get_session

if TYPE_CHECKING:
    from src.db.models import Skill

logger = logging.getLogger(__name__)

# ── Thresholds ──

ROLLBACK_REGRESSION_THRESHOLD = -0.15  # падение success_rate на 15%
ROLLBACK_MIN_SAMPLES = 10  # минимум использований для проверки
ROLLBACK_RECENT_WINDOW = 20  # окно для расчёта recent_rate


# ── Priority enum ──


class EditPriority:
    """Приоритеты правок: чем ниже число — тем важнее."""

    FAILURE_FIX = 0  # исправление ошибок
    SUCCESS_BOOST = 1  # усиление успешных паттернов
    STYLE = 2  # стилистические правки
    EXPERIMENTAL = 3  # экспериментальные

    # Keyword mapping: слово в reason → priority
    KEYWORDS: dict[int, tuple[str, ...]] = {
        FAILURE_FIX: (
            "fix",
            "bug",
            "error",
            "fail",
            "broken",
            "wrong",
            "incorrect",
            "баг",
            "ошибк",
            "сломан",
            "неправильн",
        ),
        SUCCESS_BOOST: ("improve", "boost", "optimize", "enhance", "улучш", "оптимиз"),
        STYLE: ("style", "format", "cleanup", "readability", "стил", "формат"),
        EXPERIMENTAL: ("experiment", "try", "test", "maybe", "эксперимент", "попроб"),
    }

    @classmethod
    def for_edit(cls, reason: str) -> int:
        """Определить приоритет правки по её причине."""
        r = reason.lower()
        for priority, keywords in cls.KEYWORDS.items():
            if any(kw in r for kw in keywords):
                return priority
        return cls.SUCCESS_BOOST  # default

    @classmethod
    def prioritize(cls, edits: list[dict] | list) -> list:
        """Отсортировать правки по приоритету (важные — первыми).

        Принимает list[dict] (из DB) или list[SkillEdit].
        """

        def key(edit):
            if isinstance(edit, dict):
                r = edit.get("reason", "")
            else:
                r = getattr(edit, "reason", "")
            return cls.for_edit(r)

        return sorted(edits, key=key)


# ── LR Scheduling ──


def edit_budget_for(skill_success_count: int, base_budget: int = 3) -> int:
    """Вычислить edit budget в зависимости от зрелости навыка.

    SkillOpt-inspired cosine-like decay:
    - Новые навыки (success_count < 5):  бюджет + 2  → больше экспериментов
    - Растущие (5-19):                  бюджет        → норма
    - Зрелые (20-49):                   бюджет - 1    → осторожно
    - Ветераны (50+):                   1 правка      → минимум изменений
    """
    if skill_success_count < 5:
        return base_budget + 2
    elif skill_success_count < 20:
        return base_budget
    elif skill_success_count < 50:
        return max(2, base_budget - 1)
    else:
        return 1

    return base_budget


# ── Auto-Rollback ──


async def _get_recent_success_rate(skill_id: int) -> float:
    """Вычислить success_rate за последние ROLLBACK_RECENT_WINDOW использований."""
    from src.db.models import SkillUsage

    async with get_session() as session:
        usages = await session.execute(
            select(SkillUsage.success)
            .where(
                SkillUsage.skill_id == skill_id,
                SkillUsage.success.isnot(None),
            )
            .order_by(SkillUsage.created_at.desc())
            .limit(ROLLBACK_RECENT_WINDOW)
        )
        results = [row[0] for row in usages.all()]
        if not results:
            return 1.0  # no data → assume good
        return sum(1 for r in results if r) / len(results)


async def check_and_rollback(skill: Skill) -> bool:
    """Проверить здоровье навыка и откатить к best_body при регрессе.

    Returns:
        True если был выполнен откат, False иначе.
    """
    # Skip if no best_body to rollback to
    if not skill.best_body or not skill.best_body.strip():
        return False

    # Skip if best_body equals current body (nothing to rollback to)
    if skill.best_body.strip() == (skill.body or "").strip():
        return False

    # Need enough usage data
    if (skill.success_count or 0) < ROLLBACK_MIN_SAMPLES:
        return False

    # Calculate recent success rate
    recent_rate = await _get_recent_success_rate(skill.id)

    # Calculate best success rate from edit history
    # (best_body was saved when validation_score was highest)
    best_rate = 0.7  # reasonable default for "best" version
    if hasattr(skill, "validation_score") and skill.validation_score:
        # validation_score is a quality estimate, not success rate directly
        # Use it as a rough proxy
        best_rate = max(0.5, skill.validation_score)

    # Check for regression
    if recent_rate >= best_rate * (1 + ROLLBACK_REGRESSION_THRESHOLD):
        return False  # no regression

    # Regression detected — rollback!
    logger.warning(
        "auto-rollback: skill %r (id=%d) success_rate %.2f < best %.2f (threshold %.0f%%). "
        "Rolling back to best_body.",
        skill.name,
        skill.id,
        recent_rate,
        best_rate,
        abs(ROLLBACK_REGRESSION_THRESHOLD) * 100,
    )

    from src.core.intelligence.skill_editor import bump_version

    old_version = skill.version or "1.0.0"
    skill.body = skill.best_body
    skill.version = bump_version(old_version, "patch")

    # Record rollback in history
    history = skill.edit_history_json or []
    history.append(
        {
            "op": "auto-rollback",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version_from": old_version,
            "version_to": skill.version,
            "reason": (
                f"success_rate dropped to {recent_rate:.2f} (best: {best_rate:.2f})"
            ),
        }
    )
    skill.edit_history_json = history[-20:]

    return True


async def rollback_all_regressed(owner_id: int) -> int:
    """Проверить и откатить все активные навыки с регрессом.

    Вызывается из curator_loop.

    Returns:
        Количество откаченных навыков.
    """
    from src.db.repo import list_skills, get_or_create_user

    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        skills = await list_skills(
            session, owner, enabled=True, review_status="approved", limit=200
        )

    rolled_back = 0
    for skill in skills:
        try:
            if await check_and_rollback(skill):
                rolled_back += 1
        except Exception:
            logger.exception(
                "rollback check failed for skill %r (id=%d)", skill.name, skill.id
            )

    if rolled_back:
        logger.info(
            "auto-rollback: rolled back %d skills for owner %d",
            rolled_back,
            owner_id,
        )

    return rolled_back


# ── Skill Compression ──

COMPRESS_THRESHOLD_CHARS = 2000
COMPRESS_COOLDOWN_HOURS = 168  # 7 days
COMPRESS_PROMPT = """You are a skill compressor. Given a skill body (natural language instructions for an AI agent), compress it:

RULES:
1. Keep ALL functional rules and procedures — do NOT drop any behavioral instruction.
2. Remove duplicates — if a rule is stated twice, keep the clearest version.
3. Merge similar rules — combine variations of the same instruction.
4. Shorten verbose explanations — keep the core instruction, drop examples unless critical.
5. Target: 30% shorter, but NEVER at the cost of losing a behavioral rule.
6. Return ONLY the compressed body, no markdown, no explanations.

Original body:
{body}

Compressed body:"""


async def compress_skill_body(skill, session, provider) -> tuple[bool, str]:
    """LLM-компрессия раздутого тела навыка.

    Возвращает (was_compressed: bool, new_body: str).
    Если компрессия не нужна — возвращает (False, skill.body).
    """
    body = skill.body or ""

    # Порог: не сжимаем короткие навыки
    if len(body) < COMPRESS_THRESHOLD_CHARS:
        return False, body

    # Кулдаун: не сжимаем чаще раза в неделю
    last_compressed = getattr(skill, "last_compressed_at", None)
    if last_compressed:
        if isinstance(last_compressed, str):
            last_compressed = datetime.fromisoformat(
                last_compressed.replace("Z", "+00:00")
            )
        hours_since = (
            datetime.now(timezone.utc) - last_compressed.replace(tzinfo=timezone.utc)
        ).total_seconds() / 3600
        if hours_since < COMPRESS_COOLDOWN_HOURS:
            return False, body

    # LLM-компрессия
    try:
        from src.llm.base import ChatMessage

        prompt = COMPRESS_PROMPT.format(body=body[:4000])
        response = await provider.chat(
            [ChatMessage(role="user", content=prompt)], heavy=False
        )
        compressed = response.strip()

        # Проверка: сжатое тело должно быть короче и не пустым
        if compressed and len(compressed) < len(body) * 0.95:
            return True, compressed
        return False, body
    except Exception:
        logger.warning("Skill compression failed for %s", skill.name, exc_info=True)
        return False, body
