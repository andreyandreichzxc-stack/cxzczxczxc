"""Smart AutoRouter — оркестратор, сам решает как обработать запрос."""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class RoutePurpose(str, enum.Enum):
    MAIN = "main"
    DRAFT = "draft"
    MEMORY = "memory"
    BACKGROUND = "background"
    SEARCH = "search"
    ANALYSIS = "analysis"
    URGENT = "urgent"
    FALLBACK = "fallback"


class RiskLevel(str, enum.Enum):
    LOW = "low"  # болтовня, привет
    MEDIUM = "medium"  # задача, поиск
    HIGH = "high"  # отправка сообщения, настройки
    CRITICAL = "critical"  # удаление данных, конфликты


@dataclass
class RouterTask:
    user_text: str
    purpose: RoutePurpose = RoutePurpose.MAIN
    risk: RiskLevel = RiskLevel.LOW
    need_agents: list[str] = field(default_factory=list)
    heavy: bool = False
    cache_ttl: int = 0
    max_tokens: int = 2000
    fallback_purpose: RoutePurpose | None = RoutePurpose.FALLBACK
    meta: dict = field(default_factory=dict)


@dataclass
class RouterPlan:
    tasks: list[RouterTask] = field(default_factory=list)
    final_response: str = ""
    used_providers: list[str] = field(default_factory=list)
    elapsed_ms: int = 0


async def classify_risk(user_text: str) -> RiskLevel:
    """Быстрая эвристика для определения уровня риска."""
    t = user_text.lower().strip()
    # CRITICAL: удаление, сброс
    if any(w in t for w in ("удали", "забудь", "сбрось", "очисти", "отмени всё")):
        return RiskLevel.CRITICAL
    # HIGH: отправка, настройки
    if any(
        w in t
        for w in (
            "отправь",
            "напиши",
            "скажи",
            "настрой",
            "измени",
            "включи",
            "выключи",
        )
    ):
        return RiskLevel.HIGH
    # MEDIUM: поиск, анализ
    if any(
        w in t for w in ("найди", "поищи", "анализ", "сводка", "статистика", "сколько")
    ):
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


async def make_plan(
    user_text: str,
    telegram_id: int,
    *,
    provider_available: bool = True,
    heavy_available: bool = True,
    last_purpose: str | None = None,
) -> RouterPlan:
    """
    Строит план обработки запроса.
    Автоматически выбирает: purpose, heavy/light, агентов, кэш.
    """
    risk = await classify_risk(user_text)
    plan = RouterPlan()
    start = time.monotonic()
    meta: dict[str, Any] = {}

    # ---------- Слой 1: Recall-first ----------
    try:
        from src.core.memory_recall import recall

        recall_result = await recall(
            telegram_id,
            query=user_text[:200],
            limit=3,
            include_self=True,
            include_pinned=True,
            include_tasks=True,
        )
        if recall_result and recall_result.facts:
            meta["recall_hit"] = len(recall_result.facts)
            meta["recall_facts"] = [rf.fact[:80] for rf in recall_result.facts[:3]]
    except Exception:
        pass

    # ---------- Слой 4: Context chain (last_purpose) ----------
    if (
        last_purpose
        and len(user_text) < 30
        and any(w in user_text.lower() for w in ("а ещё", "и", "тоже", "также"))
    ):
        meta["context_chain"] = last_purpose
        task = RouterTask(
            user_text,
            purpose=RoutePurpose(last_purpose),
            risk=risk,
            heavy=False,
        )
        plan.tasks.append(task)
        plan.elapsed_ms = int((time.monotonic() - start) * 1000)
        return plan

    t = user_text.lower().strip()

    # Приветствие / болтовня
    if any(
        w == t or t.startswith(w)
        for w in ("привет", "здаров", "хай", "ку", "доброе", "как дела", "чё как")
    ):
        task = RouterTask(
            user_text,
            purpose=RoutePurpose.MAIN,
            risk=RiskLevel.LOW,
            heavy=False,
            cache_ttl=60,
        )
        plan.tasks.append(task)
        plan.elapsed_ms = int((time.monotonic() - start) * 1000)
        return plan

    # ---------- Слой 2: Contact-aware routing ----------
    try:
        from src.core.contact_resolver import resolve
        from src.db.session import get_session
        from src.db.repo import get_or_create_user, get_conversation_state
        import re

        names = re.findall(
            r"(?:скажи|напиши|отправь|передай|ответь)\s+(\S+)",
            user_text,
            re.IGNORECASE,
        )
        if names:
            async with get_session() as session:
                owner = await get_or_create_user(session, telegram_id)
                contacts = await resolve(None, owner, names[0])
                if contacts and contacts[0].score >= 70:
                    peer_id = contacts[0].peer_id
                    state = await get_conversation_state(session, owner, peer_id)
                    if state and state.status == "waiting_reply":
                        meta["contact_status"] = "waiting_reply"
                        meta["contact_id"] = peer_id
                        meta["contact_name"] = contacts[0].display_name
                        task = RouterTask(
                            user_text,
                            purpose=RoutePurpose.DRAFT,
                            risk=RiskLevel.HIGH,
                            need_agents=["search", "draft"],
                        )
                        plan.tasks.append(task)
                        plan.elapsed_ms = int((time.monotonic() - start) * 1000)
                        return plan
    except Exception:
        pass

    # Отправка сообщения
    if any(w in t for w in ("отправь", "напиши", "скажи", "передай", "ответь")):
        task = RouterTask(
            user_text,
            purpose=RoutePurpose.MAIN,
            risk=RiskLevel.HIGH,
            need_agents=["search", "draft"],
            heavy=False,
            cache_ttl=0,
        )
        plan.tasks.append(task)
        plan.elapsed_ms = int((time.monotonic() - start) * 1000)
        return plan

    # Поиск
    if any(w in t for w in ("найди", "поищи")):
        task = RouterTask(
            user_text,
            purpose=RoutePurpose.SEARCH,
            risk=RiskLevel.MEDIUM,
            need_agents=["search"],
            heavy=heavy_available,
            cache_ttl=300,
        )
        plan.tasks.append(task)
        plan.elapsed_ms = int((time.monotonic() - start) * 1000)
        return plan

    # Анализ / сводка
    if any(w in t for w in ("анализ", "сводка", "проанализируй")):
        task = RouterTask(
            user_text,
            purpose=RoutePurpose.ANALYSIS,
            risk=RiskLevel.MEDIUM,
            need_agents=["search"],
            heavy=heavy_available,
            cache_ttl=300,
        )
        plan.tasks.append(task)
        plan.elapsed_ms = int((time.monotonic() - start) * 1000)
        return plan

    # Черновик ответа
    if any(w in t for w in ("напиши ответ", "черновик", "draft", "набросай ответ")):
        task = RouterTask(
            user_text,
            purpose=RoutePurpose.DRAFT,
            risk=RiskLevel.MEDIUM,
            need_agents=["draft"],
            heavy=False,
            cache_ttl=0,
        )
        plan.tasks.append(task)
        plan.elapsed_ms = int((time.monotonic() - start) * 1000)
        return plan

    # Задачи / напоминания
    if any(w in t for w in ("напомни", "задача", "дедлайн", "обещание", "план")):
        task = RouterTask(
            user_text,
            purpose=RoutePurpose.MAIN,
            risk=RiskLevel.MEDIUM,
            need_agents=["commitment"],
            heavy=False,
            cache_ttl=0,
        )
        plan.tasks.append(task)
        plan.elapsed_ms = int((time.monotonic() - start) * 1000)
        return plan

    # ---------- Слой 3: Urgency escalation ----------
    try:
        from src.core.conflict_predictor import detect_silence_triggers

        triggers = await detect_silence_triggers(telegram_id)
        if triggers:
            risky_names = [t["contact_name"].lower() for t in triggers]
            if any(name in user_text.lower() for name in risky_names):
                if risk == RiskLevel.LOW:
                    risk = RiskLevel.MEDIUM
                    meta["escalated_from"] = "low"
                elif risk == RiskLevel.MEDIUM:
                    risk = RiskLevel.HIGH
                    meta["escalated_from"] = "medium"
                meta["conflict_risk"] = [t["contact_name"] for t in triggers[:2]]
    except Exception:
        pass

    # Всё остальное → MAIN + light
    task = RouterTask(
        user_text, purpose=RoutePurpose.MAIN, risk=risk, heavy=False, cache_ttl=0
    )

    # ---------- Слой 5: Heavy/Light intelligence ----------
    should_use_heavy = heavy_available and bool(
        risk in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        or len(user_text) > 200
        or meta.get("recall_hit", 0) > 5
        or meta.get("conflict_risk")
    )
    task.heavy = should_use_heavy
    task.meta = meta

    plan.tasks.append(task)
    plan.elapsed_ms = int((time.monotonic() - start) * 1000)
    return plan
