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
) -> RouterPlan:
    """
    Строит план обработки запроса.
    Автоматически выбирает: purpose, heavy/light, агентов, кэш.
    """
    risk = await classify_risk(user_text)
    plan = RouterPlan()
    start = time.monotonic()

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

    # Поиск / анализ
    if any(w in t for w in ("найди", "поищи", "анализ", "сводка", "проанализируй")):
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

    # Всё остальное → MAIN + light
    task = RouterTask(
        user_text, purpose=RoutePurpose.MAIN, risk=risk, heavy=False, cache_ttl=0
    )
    plan.tasks.append(task)
    plan.elapsed_ms = int((time.monotonic() - start) * 1000)
    return plan
