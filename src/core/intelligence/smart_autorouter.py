"""Smart AutoRouter — оркестратор, сам решает как обработать запрос."""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Предкомпилированные константы для быстрого роутинга (не создаются заново на каждый запрос)
import re as _re

from src.db.repo import get_or_create_user, get_conversation_state
from src.db.session import get_session

_RE_CONTACT_ACTION = _re.compile(
    r"(?:скажи|напиши|отправь|передай|ответь)\s+(\S+)", _re.IGNORECASE
)
_CHAIN_WORDS = ("а ещё", "и", "тоже", "также")
_GREETINGS = ("привет", "здаров", "хай", "ку", "доброе", "как дела", "чё как")
_SEND_WORDS = ("отправь", "напиши", "скажи", "передай", "ответь")
_SEARCH_WORDS = ("найди", "поищи")
_ANALYSIS_WORDS = ("анализ", "сводка", "проанализируй")
_DRAFT_WORDS = ("напиши ответ", "черновик", "draft", "набросай ответ")
_REMINDER_WORDS = ("напомни", "задача", "дедлайн", "обещание", "план")


def _get_active_telethon_client(telegram_id: int):
    """Получить активный Telethon-клиент из синглтона UserbotManager."""
    from src.userbot import get_active_telethon_client

    return get_active_telethon_client(telegram_id)


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


class ResponseMode(str, enum.Enum):
    INSTANT = "instant"
    FAST_ROUTE = "fast_route"
    MAESTRO = "maestro"


INSTANT_PATTERNS = [
    _re.compile(
        r"^(привет|здаров|хай|ку|hello|hi|доброе утро|добрый вечер|добрый день)\b"
    ),
    _re.compile(r"^(как дела|чё как|как ты|как сам)\b"),
    _re.compile(r"^(спокойной ночи|пока|до завтра|ладн[оа])\b"),
    _re.compile(r"^(спасибо|благодарю|спс|thx)\b"),
    _re.compile(r"^(ясно|понял|ок|окей|ага|угу|ладно)\b"),
]
_HEAVY_WORDS = (
    "анализ",
    "сводка",
    "найди все",
    "проанализируй",
    "расскажи подробно",
)

INSTANT_REPLIES = {
    "привет": "Привет! 👋",
    "здаров": "Здаров! 😎",
    "хай": "Хай! ✌️",
    "ку": "Ку! 👋",
    "доброе утро": "Доброе утро! ☀️",
    "добрый вечер": "Добрый вечер! 🌆",
    "добрый день": "Добрый день! ☀️",
    "как дела": "Всё отлично! Работаю над твоими задачами 💪",
    "чё как": "Да норм! А у тебя? 😄",
    "как ты": "Я в порядке! Работаю в штатном режиме 🤖",
    "спокойной ночи": "Спокойной ночи! Сладких снов 😴🌙",
    "пока": "Пока! До связи 👋",
    "до завтра": "До завтра! 🌙",
    "спасибо": "Всегда пожалуйста! 🤗",
    "спс": "Не за что! 💪",
    "ясно": "👍",
    "понял": "✅",
    "ок": "👌",
    "окей": "👌",
    "ага": "😄",
    "ладно": "👌",
}


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
    # NEW: pre-built context
    memory_context: str = ""  # готовый memory_recall_context для инжекта
    recall_result: Any = None  # RecallResult (lazy import)
    self_profile: str = ""  # SelfProfile JSON или текст
    contact_profile: str = ""  # ContactProfile для упомянутого контакта
    rag_context: str = ""  # RAG-контекст
    response_mode: str = "maestro"  # instant | fast_route | maestro
    recall_mode: str = "deep"  # light | normal | deep
    metrics: dict = field(
        default_factory=dict
    )  # {recall_ms, router_ms, llm_ms, maestro_ms, total_ms}


async def classify_mode(user_text: str) -> ResponseMode:
    """Определяет режим ответа: instant / fast_route / maestro."""
    t = user_text.lower().strip()
    if len(t) < 30:
        for pattern in INSTANT_PATTERNS:
            m = _re.match(pattern, t)
            if m:
                return ResponseMode.INSTANT
    if len(t) < 100 and not any(w in t for w in _HEAVY_WORDS):
        return ResponseMode.FAST_ROUTE
    return ResponseMode.MAESTRO


def get_instant_reply(user_text: str) -> str:
    """Возвращает мгновенный ответ для простых фраз."""
    t = user_text.lower().strip().rstrip(".!?")
    if t in INSTANT_REPLIES:
        return INSTANT_REPLIES[t]
    for key, reply in INSTANT_REPLIES.items():
        if t.startswith(key):
            return reply
    return ""


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
    Определяет режим ответа, собирает контекст (recall, self_profile),
    выбирает purpose, heavy/light, агентов, кэш.
    """
    plan = RouterPlan()
    start = time.monotonic()

    # Шаг 0: определить режим ответа
    mode = await classify_mode(user_text)
    plan.response_mode = mode.value
    plan.metrics["mode"] = mode.value

    if mode == ResponseMode.INSTANT:
        plan.final_response = get_instant_reply(user_text)
        plan.tasks = []
        plan.elapsed_ms = int((time.monotonic() - start) * 1000)
        plan.metrics["total_ms"] = plan.elapsed_ms
        return plan

    risk = await classify_risk(user_text)
    meta: dict[str, Any] = {}

    # ---------- Слой 1: Memory recall (один раз!) ----------
    try:
        from src.core.memory.conversation_depth import decide_context_depth

        depth_decision = decide_context_depth(telegram_id, user_text)
        plan.recall_mode = depth_decision.recall_mode
        plan.metrics["dialog_depth"] = depth_decision.depth
        plan.metrics["message_weight"] = round(depth_decision.message_weight, 2)
        plan.metrics["recall_mode"] = depth_decision.recall_mode
    except Exception:
        plan.recall_mode = "deep"

    t1 = time.monotonic()
    try:
        from src.core.memory.memory_recall import recall, format_recall_for_prompt

        recall_result = await recall(
            telegram_id,
            query=user_text[:200],
            limit=10,
            include_self=True,
            include_pinned=True,
            include_tasks=True,
            include_deep=plan.recall_mode == "deep",
            mode=plan.recall_mode,
        )
        plan.recall_result = recall_result
        plan.memory_context = format_recall_for_prompt(recall_result)
        if recall_result and recall_result.facts:
            meta["recall_hit"] = len(recall_result.facts)
            meta["recall_facts"] = [rf.fact[:80] for rf in recall_result.facts[:3]]
    except Exception:
        logger.debug("smart_autorouter: recall skipped", exc_info=True)
        pass
    plan.metrics["recall_ms"] = int((time.monotonic() - t1) * 1000)

    # ---------- Шаг 2: Self-profile (саморефлексия) ----------
    try:
        from src.db.repo import get_self_profile

        async with get_session() as session:
            owner = await get_or_create_user(session, telegram_id)
            sp = await get_self_profile(session, owner)
            if sp:
                parts = []
                if sp.preferences:
                    parts.append(f"Предпочтения: {sp.preferences}")
                if sp.goals:
                    parts.append(f"Цели: {sp.goals}")
                plan.self_profile = "; ".join(parts)
    except Exception:
        logger.debug("smart_autorouter: self-profile skipped", exc_info=True)
        pass

    # ---------- Слой 3: Context chain (last_purpose) ----------
    if (
        last_purpose
        and len(user_text) < 30
        and any(w in user_text.lower() for w in _CHAIN_WORDS)
    ):
        meta["context_chain"] = last_purpose
        task = RouterTask(
            user_text,
            purpose=RoutePurpose(last_purpose),
            risk=risk,
            heavy=False,
        )
        plan.tasks.append(task)
        plan.metrics["router_ms"] = int((time.monotonic() - start) * 1000)
        plan.elapsed_ms = plan.metrics["router_ms"]
        return plan

    t = user_text.lower().strip()

    # Приветствие / болтовня
    if any(w == t or t.startswith(w) for w in _GREETINGS):
        task = RouterTask(
            user_text,
            purpose=RoutePurpose.MAIN,
            risk=RiskLevel.LOW,
            heavy=False,
            cache_ttl=60,
        )
        plan.tasks.append(task)
        plan.metrics["router_ms"] = int((time.monotonic() - start) * 1000)
        plan.elapsed_ms = plan.metrics["router_ms"]
        return plan

    # ---------- Слой 4: Contact-aware routing ----------
    try:
        from src.core.contacts.contact_resolver import resolve

        names = _RE_CONTACT_ACTION.findall(user_text)
        if names:
            async with get_session() as session:
                owner = await get_or_create_user(session, telegram_id)
                contacts = await resolve(
                    _get_active_telethon_client(telegram_id), owner, names[0]
                )
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
                        plan.metrics["router_ms"] = int(
                            (time.monotonic() - start) * 1000
                        )
                        plan.elapsed_ms = plan.metrics["router_ms"]
                        return plan
    except Exception:
        logger.debug("smart_autorouter: contact routing skipped", exc_info=True)
        pass

    # Отправка сообщения
    if any(w in t for w in _SEND_WORDS):
        task = RouterTask(
            user_text,
            purpose=RoutePurpose.MAIN,
            risk=RiskLevel.HIGH,
            need_agents=["search", "draft"],
            heavy=False,
            cache_ttl=0,
        )
        plan.tasks.append(task)
        plan.metrics["router_ms"] = int((time.monotonic() - start) * 1000)
        plan.elapsed_ms = plan.metrics["router_ms"]
        return plan

    # Поиск
    if any(w in t for w in _SEARCH_WORDS):
        task = RouterTask(
            user_text,
            purpose=RoutePurpose.SEARCH,
            risk=RiskLevel.MEDIUM,
            need_agents=["search"],
            heavy=heavy_available,
            cache_ttl=300,
        )
        plan.tasks.append(task)
        plan.metrics["router_ms"] = int((time.monotonic() - start) * 1000)
        plan.elapsed_ms = plan.metrics["router_ms"]
        return plan

    # Анализ / сводка
    if any(w in t for w in _ANALYSIS_WORDS):
        task = RouterTask(
            user_text,
            purpose=RoutePurpose.ANALYSIS,
            risk=RiskLevel.MEDIUM,
            need_agents=["search"],
            heavy=heavy_available,
            cache_ttl=300,
        )
        plan.tasks.append(task)
        plan.metrics["router_ms"] = int((time.monotonic() - start) * 1000)
        plan.elapsed_ms = plan.metrics["router_ms"]
        return plan

    # Черновик ответа
    if any(w in t for w in _DRAFT_WORDS):
        task = RouterTask(
            user_text,
            purpose=RoutePurpose.DRAFT,
            risk=RiskLevel.MEDIUM,
            need_agents=["draft"],
            heavy=False,
            cache_ttl=0,
        )
        plan.tasks.append(task)
        plan.metrics["router_ms"] = int((time.monotonic() - start) * 1000)
        plan.elapsed_ms = plan.metrics["router_ms"]
        return plan

    # Задачи / напоминания
    if any(w in t for w in _REMINDER_WORDS):
        task = RouterTask(
            user_text,
            purpose=RoutePurpose.MAIN,
            risk=RiskLevel.MEDIUM,
            need_agents=["commitment"],
            heavy=False,
            cache_ttl=0,
        )
        plan.tasks.append(task)
        plan.metrics["router_ms"] = int((time.monotonic() - start) * 1000)
        plan.elapsed_ms = plan.metrics["router_ms"]
        return plan

    # ---------- Слой 5: Urgency escalation ----------
    try:
        from src.core.actions.conflict_predictor import detect_silence_triggers

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
        logger.debug("smart_autorouter: intent classification skipped", exc_info=True)
        pass

    # Всё остальное → MAIN + light
    task = RouterTask(
        user_text, purpose=RoutePurpose.MAIN, risk=risk, heavy=False, cache_ttl=0
    )

    # ---------- Слой 6: Heavy/Light intelligence ----------
    should_use_heavy = heavy_available and bool(
        risk in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        or len(user_text) > 200
        or meta.get("recall_hit", 0) > 5
        or meta.get("conflict_risk")
    )
    task.heavy = should_use_heavy
    task.meta = meta

    plan.tasks.append(task)
    plan.metrics["router_ms"] = int((time.monotonic() - start) * 1000)
    plan.elapsed_ms = plan.metrics["router_ms"]
    return plan
