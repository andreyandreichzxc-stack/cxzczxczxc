"""RouterPlan + make_plan — построение плана обработки запроса."""

from __future__ import annotations

import logging
import re as _re
import time
from dataclasses import dataclass, field
from typing import Any

from src.db.repo import get_or_create_user, get_conversation_state
from src.db.session import get_session

from ..pattern_cache import pattern_cache
from ..routing_wordlists import (
    CHAIN_WORDS,
    GREETINGS,
    SEND_WORDS,
    SEARCH_WORDS,
    ANALYSIS_WORDS,
    DRAFT_WORDS,
    REMINDER_WORDS,
    PERSON_INFO_WORDS,
    CONTACT_ACTION_PATTERN,
    PERSON_INFO_PATTERN,
    GENERIC_NAME_PATTERN,
    learned_match,
)
from .classifier import (
    RoutePurpose,
    RiskLevel,
    ResponseMode,
    _LEARNED_TASK_MAP,
    classify_mode,
    classify_risk,
    get_instant_reply,
)

logger = logging.getLogger(__name__)

_RE_CONTACT_ACTION = _re.compile(CONTACT_ACTION_PATTERN, _re.IGNORECASE)
_RE_PERSON_INFO = _re.compile(PERSON_INFO_PATTERN, _re.IGNORECASE)
_RE_GENERIC_NAME = _re.compile(GENERIC_NAME_PATTERN)


def _get_active_telethon_client(telegram_id: int):
    """Получить активный Telethon-клиент из синглтона UserbotManager."""
    from src.userbot import get_active_telethon_client

    return get_active_telethon_client(telegram_id)


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

        depth_decision = await decide_context_depth(telegram_id, user_text)
        plan.recall_mode = depth_decision.recall_mode
        plan.metrics["dialog_depth"] = depth_decision.depth
        plan.metrics["message_weight"] = round(depth_decision.message_weight, 2)
        plan.metrics["recall_mode"] = depth_decision.recall_mode
    except Exception:
        plan.recall_mode = "deep"

    t1 = time.monotonic()
    # ---------- Общая сессия для recall + self-profile ----------
    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)

        # Recall с общей сессией
        try:
            from src.core.memory.memory_recall import recall, format_recall_for_prompt

            recall_result = await recall(
                telegram_id,
                session=session,
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

        # Self-profile той же сессией
        try:
            from src.db.repo import get_self_profile

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
    plan.metrics["recall_ms"] = int((time.monotonic() - t1) * 1000)

    # ---------- Слой 3: Context chain (last_purpose) ----------
    if (
        last_purpose is not None
        and len(user_text) < 30
        and any(w in user_text.lower() for w in CHAIN_WORDS)
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

    # Слой 3b: Learned routing (self-improving keyword router)
    learned_intent = learned_match(t)
    if learned_intent is not None:
        logger.debug("Learned routing matched: '%s' → %s", user_text, learned_intent)

        # Feature 2: Pattern cache — проверяем закэшированный action перед LLM
        try:
            async with get_session() as _pc_session:
                _pc_owner = await get_or_create_user(_pc_session, telegram_id)
                if _pc_owner.settings and _pc_owner.settings.pattern_caching_enabled:
                    cached_action = await pattern_cache.get_cached_action(
                        telegram_id, learned_intent
                    )
                    if cached_action is not None:
                        logger.debug(
                            "Pattern cache BYPASS for user=%d intent=%s → %s",
                            telegram_id,
                            learned_intent,
                            cached_action,
                        )
                        task_spec = _LEARNED_TASK_MAP.get(cached_action)
                        if task_spec is not None:
                            purpose, risk, need_agents, heavy, cache_ttl = task_spec
                            task = RouterTask(
                                user_text,
                                purpose=RoutePurpose(purpose),
                                risk=RiskLevel(risk),
                                need_agents=list(need_agents),
                                heavy=heavy and heavy_available,
                                cache_ttl=cache_ttl,
                            )
                            task.meta = {
                                "pattern_cache_hit": True,
                                "intent": learned_intent,
                            }
                            plan.tasks.append(task)
                            plan.metrics["router_ms"] = int(
                                (time.monotonic() - start) * 1000
                            )
                            plan.elapsed_ms = plan.metrics["router_ms"]
                            plan.metrics["pattern_cache"] = "hit"
                            return plan
        except Exception:
            logger.debug("Pattern cache check skipped", exc_info=True)

        task_spec = _LEARNED_TASK_MAP.get(learned_intent)
        if task_spec is not None:
            purpose, risk, need_agents, heavy, cache_ttl = task_spec
            task = RouterTask(
                user_text,
                purpose=RoutePurpose(purpose),
                risk=RiskLevel(risk),
                need_agents=list(need_agents),
                heavy=heavy and heavy_available,
                cache_ttl=cache_ttl,
            )
            plan.tasks.append(task)
            plan.metrics["router_ms"] = int((time.monotonic() - start) * 1000)
            plan.elapsed_ms = plan.metrics["router_ms"]
            return plan

    # Приветствие / болтовня
    if any(w == t or t.startswith(w) for w in GREETINGS):
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
    if any(w in t for w in SEND_WORDS):
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
    if any(w in t for w in SEARCH_WORDS):
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
    if any(w in t for w in ANALYSIS_WORDS):
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
    if any(w in t for w in DRAFT_WORDS):
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
    if any(w in t for w in REMINDER_WORDS):
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

    # ---------- Слой 4.5: Person-info contact resolution ----------
    # Определяет запросы типа «как тебе Влад?», «что думаешь о Насте?»,
    # «расскажи про Колю», «какой человек Саша?» и т.д.
    # Если найден контакт — делает таргетированный memory recall,
    # подтягивает живой контекст (последние сообщения) и обогащает контекст.
    _person_info_enriched = False
    try:
        # Шаг 1: ищем имя через PERSON_INFO_PATTERN (явные вопросы)
        person_names = _RE_PERSON_INFO.findall(user_text)
        if not person_names:
            # Шаг 2: если запрос содержит PERSON_INFO_WORDS — ищем имя через generic pattern
            t_lower = user_text.lower()
            if any(piw in t_lower for piw in PERSON_INFO_WORDS):
                person_names = _RE_GENERIC_NAME.findall(user_text)

        if person_names:
            async with get_session() as session:
                owner = await get_or_create_user(session, telegram_id)

                # ── Кэш контакт-резолвинга (5 мин) ──
                cache_key = f"contact_resolve:{telegram_id}:{person_names[0].lower()}"
                contacts = None
                try:
                    from src.core.context_cache import (
                        get as cache_get,
                        put as cache_put,
                    )

                    contacts = await cache_get(cache_key)
                except Exception:
                    pass

                if contacts is None:
                    client = _get_active_telethon_client(telegram_id)
                    if client is not None:
                        contacts = await resolve(client, owner, person_names[0])
                        if contacts:
                            try:
                                await cache_put(cache_key, contacts, ttl=300)
                            except Exception:
                                pass

                if contacts and contacts[0].score >= 60:
                    peer_id = contacts[0].peer_id
                    contact_name = contacts[0].display_name
                    meta["person_info_contact_id"] = peer_id
                    meta["person_info_contact_name"] = contact_name
                    meta["person_info_score"] = contacts[0].score

                    enriched_parts: list[str] = []

                    # ── Таргетированный memory recall для контакта ──
                    try:
                        from src.core.memory.memory_recall import (
                            recall,
                            format_recall_for_prompt,
                        )

                        contact_recall = await recall(
                            telegram_id,
                            session=session,
                            contact_id=peer_id,
                            query=user_text[:200],
                            limit=10,
                            include_self=True,
                            include_pinned=True,
                            include_tasks=False,
                            include_deep=True,
                            mode=plan.recall_mode,
                        )
                        if contact_recall and contact_recall.facts:
                            contact_context = format_recall_for_prompt(contact_recall)
                            if contact_context:
                                enriched_parts.append(
                                    f"## ФАКТЫ О КОНТАКТЕ «{contact_name}»:\n"
                                    f"{contact_context}"
                                )
                                meta["person_info_recall_hit"] = len(
                                    contact_recall.facts
                                )
                                _person_info_enriched = True
                                logger.debug(
                                    "Person-info: resolved %r → %s, recalled %d facts",
                                    person_names[0],
                                    contact_name,
                                    len(contact_recall.facts),
                                )
                    except Exception:
                        logger.debug(
                            "Person-info: recall skipped for %r",
                            person_names[0],
                            exc_info=True,
                        )

                    # ── Живой контекст: последние сообщения с контактом ──
                    try:
                        from src.db.repo import fetch_chat_messages
                        from src.core.contacts.chat_service import (
                            messages_to_transcript,
                        )

                        recent_msgs = await fetch_chat_messages(
                            session, owner, peer_id, limit=15
                        )
                        if recent_msgs:
                            transcript = messages_to_transcript(recent_msgs)
                            # Обрезаем до ~1500 символов чтобы не перегружать контекст
                            if len(transcript) > 1500:
                                transcript = transcript[-1500:]
                                transcript = (
                                    "…(ранние сообщения опущены)\n" + transcript
                                )
                            enriched_parts.append(
                                f"## ПОСЛЕДНЯЯ ПЕРЕПИСКА С «{contact_name}»:\n"
                                f"```\n{transcript}\n```"
                            )
                            meta["person_info_live_msgs"] = len(recent_msgs)
                            logger.debug(
                                "Person-info: fetched %d live messages for %r",
                                len(recent_msgs),
                                contact_name,
                            )
                    except Exception:
                        logger.debug(
                            "Person-info: live messages skipped for %r",
                            person_names[0],
                            exc_info=True,
                        )

                    # ── Объединяем обогащённый контекст ──
                    if enriched_parts:
                        existing = plan.memory_context or ""
                        plan.memory_context = (
                            "\n\n".join(enriched_parts) + "\n\n" + existing
                        )
    except Exception:
        logger.debug("smart_autorouter: person-info routing skipped", exc_info=True)

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
