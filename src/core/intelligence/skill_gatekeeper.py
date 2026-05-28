"""Skill Gatekeeper — бесплатный pre-filter перед анализом навыков.  0 токенов.

Решает, СТОИТ ЛИ вообще запускать LLM-анализ для предложения новых навыков
или правок.  Отсекает ~90% бесполезных вызовов.

Принципы:
- Не анализируй, если нечего (хеш сообщений не менялся)
- Не анализируй слишком часто (cooldown)
- Не анализируй без достаточных данных (мало траекторий)

Все проверки — чистые вычисления, без LLM-вызовов.
"""

from __future__ import annotations

import hashlib
import logging
import json
import re
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from sqlalchemy import func, select

from src.db.session import get_session

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ── Thresholds (tunable via config or direct override) ──

MSG_DELTA_MIN = 20  # минимум +20 новых сообщений с прошлого анализа
COOLDOWN_HOURS = 6  # не анализировать чаще, чем раз в N часов
MIN_TRAJECTORIES = 5  # минимум 5 новых траекторий с прошлого анализа
MIN_DAYS_SINCE_START = 1  # не анализировать в первый день работы (мало данных)

# Key for storing last analysis snapshot in DB (via user metadata or cache)
CACHE_NAMESPACE = "skill_gatekeeper"


def _hash_messages(messages: list[str]) -> str:
    """Быстрый хеш списка сообщений (sha256, первые 16 символов)."""
    concat = "\n".join(sorted(m.lower() for m in messages if m))
    return hashlib.sha256(concat.encode()).hexdigest()[:16]


# ── Gatekeeper ────────────────────────────────────────────────────────


class SkillGatekeeper:
    """Pre-filter для LLM-анализа навыков.

    Использование:
        gatekeeper = SkillGatekeeper()
        if await gatekeeper.should_analyze(owner_id):
            await propose_skills_from_analysis(...)  # только если нужно
    """

    def __init__(self) -> None:
        # In-memory snapshot: {owner_id: {"hash": str, "ts": float, "traj_count": int}}
        self._snapshots: dict[int, dict] = {}

    async def _get_recent_messages(self, owner_id: int) -> list[str]:
        """Получить последние N сообщений владельца для хеширования."""
        from src.db.repo import fetch_my_messages_global
        from src.db.repo import get_or_create_user

        async with get_session() as session:
            owner = await get_or_create_user(session, owner_id)
            messages_raw = await fetch_my_messages_global(session, owner, limit=50)
            return [msg.text or "" for msg in messages_raw if msg.text]

    async def _get_recent_trajectory_count(self, owner_id: int) -> int:
        """Количество успешных траекторий за последние COOLDOWN_HOURS часов."""
        from src.db.models import Trajectory

        since = datetime.now(timezone.utc) - timedelta(hours=COOLDOWN_HOURS)
        async with get_session() as session:
            result = await session.execute(
                select(func.count(Trajectory.id)).where(
                    Trajectory.user_id == owner_id,
                    Trajectory.success.is_(True),
                    Trajectory.created_at >= since,
                )
            )
            return result.scalar() or 0

    async def _get_earliest_message_date(self, owner_id: int) -> datetime | None:
        """Дата самого раннего сообщения пользователя (для проверки MIN_DAYS_SINCE_START)."""
        from src.db.models import Trajectory

        async with get_session() as session:
            result = await session.execute(
                select(Trajectory.created_at)
                .where(Trajectory.user_id == owner_id)
                .order_by(Trajectory.created_at.asc())
                .limit(1)
            )
            row = result.scalar_one_or_none()
            return row

    async def should_analyze(self, owner_id: int) -> tuple[bool, str]:
        """Проверить, стоит ли запускать LLM-анализ.

        Returns:
            (should_run, reason) — где reason объясняет решение.
        """
        # Check 0: прошло ли MIN_DAYS_SINCE_START с первого сообщения?
        earliest = await self._get_earliest_message_date(owner_id)
        if earliest is not None:
            days_since_start = (datetime.now(timezone.utc) - earliest).days
            if days_since_start < MIN_DAYS_SINCE_START:
                return (
                    False,
                    f"too_early: only {days_since_start}d since first message (< {MIN_DAYS_SINCE_START}d)",
                )

        # Check 1: cooldown — не анализировать слишком часто
        now_ts = datetime.now(timezone.utc).timestamp()
        snap = self._snapshots.get(owner_id)
        if snap is not None:
            elapsed_h = (now_ts - snap["ts"]) / 3600
            if elapsed_h < COOLDOWN_HOURS:
                remaining = int(COOLDOWN_HOURS - elapsed_h)
                return (
                    False,
                    f"cooldown: {elapsed_h:.1f}h since last run (min {COOLDOWN_HOURS}h, ~{remaining}h remain)",
                )

        # Check 2: дельта сообщений — появилось ли что-то новое?
        messages = await self._get_recent_messages(owner_id)
        if not messages:
            return False, "no_messages"

        current_hash = _hash_messages(messages)

        if snap is not None and snap.get("hash") == current_hash:
            return False, "no_new_messages: message hash unchanged"

        # Check 3: достаточно ли новых траекторий?
        traj_count = await self._get_recent_trajectory_count(owner_id)
        if traj_count < MIN_TRAJECTORIES:
            return (
                False,
                f"insufficient_trajectories: {traj_count} < {MIN_TRAJECTORIES}",
            )

        # Все проверки пройдены — сохраняем снепшот
        self._snapshots[owner_id] = {
            "hash": current_hash,
            "ts": now_ts,
            "traj_count": traj_count,
        }

        return (
            True,
            f"ready: {traj_count} trajectories, {len(messages)} messages, delta={traj_count - snap.get('traj_count', 0)}",
        )

    def reset(self, owner_id: int) -> None:
        """Сбросить кэш для owner_id (принудительный анализ)."""
        self._snapshots.pop(owner_id, None)
        logger.debug("gatekeeper: reset for owner %d", owner_id)


# ── Singleton ──

_gatekeeper: SkillGatekeeper | None = None


def get_gatekeeper() -> SkillGatekeeper:
    """Получить синглтон gatekeeper."""
    global _gatekeeper
    if _gatekeeper is None:
        _gatekeeper = SkillGatekeeper()
    return _gatekeeper


# ── Utilities for tiered analysis ──


def extract_light_patterns(messages: list[str]) -> list[str] | None:
    """Бесплатный regex-анализ: ищет явные повторяющиеся паттерны.

    Ищет:
    - Повторяющиеся intent-слова (найди, напиши, отправь, покажи)
    - Повторяющиеся имена контактов (с большой буквы, 2+ букв)
    - Временные паттерны (каждый день, по понедельникам)

    Returns:
        Список найденных паттернов или None.
        Используется как hint для light-анализа.
    """
    if len(messages) < 5:
        return None

    # Intent words
    intent_words = re.compile(
        r"\b(найди|поищи|напиши|отправь|покажи|расскажи|напомни|запланируй|переведи|суммируй)\b",
        re.IGNORECASE,
    )
    intent_counts: dict[str, int] = {}
    for msg in messages:
        found = intent_words.findall(msg.lower())
        for w in found:
            intent_counts[w] = intent_counts.get(w, 0) + 1

    patterns = []
    for word, count in intent_counts.items():
        if count >= 3:
            patterns.append(f"intent:{word}")

    # Contact names (capitalized words, 3+ chars, not at sentence start)
    contact_re = re.compile(r"\b([А-ЯA-Z][а-яa-z]{2,})\b")
    contact_counts: dict[str, int] = {}
    for msg in messages:
        found = contact_re.findall(msg)
        for w in found:
            contact_counts[w] = contact_counts.get(w, 0) + 1

    for name, count in contact_counts.items():
        if count >= 3:
            patterns.append(f"contact:{name}")

    # Time patterns
    time_patterns = re.compile(
        r"\b(каждый день|по понедельникам|раз в неделю|каждый час|утром|вечером|завтра)\b",
        re.IGNORECASE,
    )
    time_count = sum(1 for msg in messages if time_patterns.search(msg.lower()))
    if time_count >= 3:
        patterns.append("time:recurring")

    return patterns if patterns else None


# ── Skill dependency/conflict detection (Feature 4) ──


def _normalize_pattern(p: str) -> str:
    """Приводит паттерн к нижнему регистру для сравнения."""
    return p.lower().strip().rstrip("$").lstrip("^")


def detect_skill_conflicts(
    new_name: str,
    new_triggers: list,
    existing_skills: list,  # list of (name, trigger_patterns_json)
    overlap_threshold: float = 0.4,
) -> list[dict]:
    """Бесплатная проверка пересечения trigger-паттернов между навыками.

    Возвращает список конфликтов: [{"skill": name, "overlap_ratio": float, "shared_patterns": [...]}, ...]
    Пустой список = нет конфликтов.
    """
    if not new_triggers:
        return []

    new_set = {
        _normalize_pattern(t) for t in new_triggers if isinstance(t, str) and t.strip()
    }
    if not new_set:
        return []

    conflicts = []
    for skill_name, triggers_json in existing_skills:
        if skill_name == new_name:
            continue

        if not triggers_json:
            continue

        try:
            existing_triggers = (
                triggers_json
                if isinstance(triggers_json, list)
                else json.loads(triggers_json)
                if isinstance(triggers_json, str)
                else []
            )
        except (json.JSONDecodeError, TypeError):
            continue

        existing_set = {
            _normalize_pattern(t)
            for t in existing_triggers
            if isinstance(t, str) and t.strip()
        }
        if not existing_set:
            continue

        shared = new_set & existing_set
        if not shared:
            soft_shared: set[str] = set()
            for np in new_set:
                for ep in existing_set:
                    if len(np) > 3 and len(ep) > 3 and (np in ep or ep in np):
                        soft_shared.add(f"{np} ≈ {ep}")
            shared = soft_shared

        if not shared:
            continue

        overlap_ratio = len(shared) / max(len(new_set), len(existing_set))
        if overlap_ratio >= overlap_threshold:
            conflicts.append(
                {
                    "skill": skill_name,
                    "overlap_ratio": round(overlap_ratio, 2),
                    "shared_patterns": list(shared)[:5],
                }
            )

    conflicts.sort(key=lambda c: c["overlap_ratio"], reverse=True)
    return conflicts


def format_conflict_warning(conflicts: list[dict]) -> str:
    """Форматирует предупреждение о конфликтах для rejected-буфера."""
    if not conflicts:
        return ""

    lines = ["⚠️ Обнаружены пересечения trigger-паттернов:"]
    for c in conflicts[:3]:
        lines.append(
            f"  - {c['skill']} (overlap: {c['overlap_ratio']:.0%}, "
            f"shared: {', '.join(c['shared_patterns'][:3])})"
        )
    return "\n".join(lines)
