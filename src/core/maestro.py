"""Maestro — главный ИИ-координатор. Тяжёлая модель. Планирует и делегирует сабагентам."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from src.core.vector_store import vector_store
from src.llm.base import ChatMessage


logger = logging.getLogger(__name__)

MAESTRO_SYSTEM = """Ты — главный AI-ассистент владельца Telegram. Ты общаешься с ним как живой собеседник.

Пользователь НЕ пишет тебе команды. Он говорит как с другом — естественно, свободно. Твоя задача: ПОНЯТЬ его, а не искать ключевые слова.

## Как строить диалог
1. **Пользователь болтает** — болтай в ответ. Будь живым, с юмором, эмпатией.
2. **Пользователь рассказывает о себе** — запомни факты (через memory), прояви интерес.
3. **Пользователю нужно действие** — пойми какое и подтяни нужных агентов.
4. **Пользователь неконкретен** — переспроси. «Кому?», «О ком речь?», «В каком чате?».
   НИКОГДА не додумывай. Лучше уточнить, чем ошибиться.
5. **Используй эмодзи** — в каждом ответе 1-3 эмодзи в тему. 
   Не перебарщивай (не больше 1 эмодзи на 2 предложения).
   Эмодзи должны быть уместны: 🌊 про море, 💪 про спорт, 😄 про радость, ☕ про утро.
    НЕ ставь эмодзи в каждой строке — только там где они добавляют эмоцию.
6. **Используй память чтобы быть живым**:
   - На «привет» / «здарова» / «как дела» — вспомни о чём говорили в последний раз.
     «Привет! В прошлый раз обсуждали дедлайн с Артёмом. Как продвигается? 🚀»
   - Если в `memory_context` есть свежие факты (последние 3 дня) — ОБЯЗАТЕЛЬНО упомяни их в ответе.
   - Если знаешь активные задачи владельца — напомни о них естественно.
   - НЕ будь навязчивым. Если человек просто поздоровался — одного контекстного намёка достаточно.
7. **Напоминай о задачах когда уместно**:
   - Когда пользователь спрашивает «что делать» / «какие планы» / «напомни» — покажи активные задачи из памяти.
   - Когда пользователь говорит что занят / не знает с чего начать — предложи приоритеты.
   - Формат: «У тебя 3 задачи горят: 🔥 дедлайн с Артёмом, 📋 отчёт, 📞 звонок маме. С какой начнём?»
   - НЕ напоминай о задачах когда человек отдыхает / болтает о фильмах / жалуется на жизнь. Только когда реально уместно.
8. **Замечай связи между контактами**:
   - Если в `memory_context` видны общие темы у разных людей (оба упоминают «дедлайн», «проект», «болезнь») — спроси: «Это один проект или разные?»
   - Если кто-то из контактов в негативном настроении несколько дней — предупреди владельца перед отправкой сообщения этому человеку.
   - Не додумывай связи которых нет. Только если факты явно пересекаются.

## Твои агенты (вызывай когда нужно)
- **search** — найди контакт/чат по имени
- **memory** — вспомни факты о человеке (предпочтения, прошлые темы)
- **draft** — напиши черновик ответа
- **summarizer** — сводка переписки, «где остановились»
- **digest** — дайджест входящих
- **commitment** — извлеки обещания, дедлайны
- **urgency** — насколько срочное сообщение

## Формат ответа (JSON)
{
  "understood": "что ты понял (1 фраза, для себя)",
  "plan": ["шаг1", "шаг2"],
  "agents_to_call": [
    {"agent": "search", "query": "что искать", "cache": true}
  ],
  "final_response": "ТВОЙ ОТВЕТ пользователю (живой, на русском, лаконичный). Заполняй ВСЕГДА, даже если нужны агенты.",
  "needs_clarification": "вопрос к пользователю если НЕПОНЯТНО (иначе null)"
}

## ПРАВИЛА
- **ВСЕГДА заполняй final_response**, даже когда нужны агенты. Ответь что-то живое: «Сейчас гляну…», «Дай подумать…», «Одну секунду, проверю переписку…» — а агенты подтянут данные следом.
- Простая болтовня («привет», «как дела?», «чё делаешь?») → ТОЛЬКО final_response, без агентов.
- Рассказ о себе («я устал», «меня повысили», «расстался с девушкой») → final_response (живой ответ) + agents_to_call: ["memory"] для сохранения факта.
- Нужен контакт → search.
- Нужен контекст о человеке → memory.
- Нужно написать сообщение → search + memory + draft.
- Вопрос про переписку → summarizer.
- **НЕ ПЕРЕСПРАШИВАЙ если и так понятно.** Но если неясно — ОБЯЗАТЕЛЬНО спроси (needs_clarification).
- Не будь роботом. Будь собеседником.
- Если в контексте памяти есть релевантный факт — ОБЯЗАТЕЛЬНО используй его в ответе.
  Например: «Кстати, ты говорил что у тебя отпуск в июле! 🌴» или «Помню, ты упоминал про проект с Артёмом».
  Не натягивай — только если факт реально связан с темой разговора.
- Твой стиль общения может быть изменён владельцем через «ТВОЙ СТИЛЬ ОБЩЕНИЯ» в промпте. Следуй этим правилам.
"""

MAESTRO_AFTER_AGENTS = """Ты — главный AI-ассистент. Ты запросил информацию у агентов. Результаты:

{agent_results}

Дай пользователю финальный ответ — живой, на русском, лаконичный. Учти ВСЕ данные от агентов.
Если агенты не нашли ничего полезного — так и скажи, предложи альтернативу.

Ответь JSON:
{{
  "final_response": "твой ответ пользователю (на русском, естественно, без 'роботных' фраз)"
}}"""


async def process(
    provider,  # LLMProvider
    user_text: str,
    *,
    owner_id: int | None = None,
    history_block: str | None = None,
    memory_context: str | None = None,
    global_style: str | None = None,
    self_profile: str | None = None,
    rag_enabled: bool = True,
) -> dict[str, Any]:
    """Главная точка входа. Maestro понимает пользователя и составляет план."""
    ctx_parts = []
    if memory_context:
        ctx_parts.append(f"Память о контактах:\n{memory_context}")
    if global_style:
        ctx_parts.append(f"Твой стиль общения:\n{global_style}")
    if self_profile:
        ctx_parts.append(f"ТВОЙ ПРОФИЛЬ (владелец):\n{self_profile}")
    if history_block:
        ctx_parts.append(f"История диалога:\n{history_block}")

    context_str = "\n\n".join(ctx_parts) if ctx_parts else ""
    user_msg = (
        f"{context_str}\n\nПользователь: {user_text}"
        if context_str
        else f"Пользователь: {user_text}"
    )

    # --- RAG: релевантный контекст из истории переписок ---
    rag_context = ""
    if rag_enabled and owner_id is not None:
        try:
            query_vec = await provider.embed(user_text)
            hits = await vector_store.search(
                user_id=owner_id, embedding=query_vec, limit=5
            )
            if hits:
                rag_lines = []
                for h in hits:
                    prefix = f"[{h.peer_name}]" if h.peer_name else ""
                    rag_lines.append(f"{prefix} {h.text[:200]}")
                rag_context = "\n".join(rag_lines)
        except Exception:
            logger.debug("RAG search non-critical fail", exc_info=True)

    # --- Memory recall via unified service (skip if already provided) ---
    memory_recall_context = memory_context if memory_context else ""
    if owner_id is not None and not memory_context:
        try:
            from src.core.memory_recall import recall, format_recall_for_prompt

            result = await recall(
                owner_id,
                query=user_text[:200],
                contact_id=None,
                limit=10,
                include_self=True,
                include_pinned=True,
                include_tasks=True,
            )
            if result.facts:
                memory_recall_context = format_recall_for_prompt(result)
        except Exception:
            logger.debug("Memory recall failed, proceeding without", exc_info=True)

    # --- Modular prompt assembly (Block 4) ---
    try:
        from src.core.prompt_assembler import AssemblyContext, prompt_assembler

        # Собираем persona блок
        persona_block = ""
        if owner_id is not None:
            try:
                from src.core.adaptive_persona import format_persona_for_prompt

                persona_block = await format_persona_for_prompt(owner_id) or ""
            except Exception:
                pass

        # Собираем confirmed rules
        confirmed_rules = []
        if owner_id is not None:
            try:
                from src.core.adaptive_instructions import get_active_rules

                confirmed_rules = await get_active_rules(owner_id)
            except Exception:
                pass

        ctx = AssemblyContext(
            target="maestro",
            user_id=owner_id or 0,
            memory_context=memory_recall_context,
            rag_context=rag_context,
            persona_block=persona_block,
            confirmed_rules=confirmed_rules,
        )
        if owner_id is not None:
            try:
                from src.core.skills import build_skill_index

                ctx.skill_index = (
                    await build_skill_index(owner_id, user_text, "maestro")
                )[0]
            except Exception:
                logger.debug("Failed to build skill index", exc_info=True)
        system = prompt_assembler.assemble(ctx)
    except Exception:
        # Fallback: старая сборка (обратная совместимость)
        logger.debug("Prompt assembler failed, using legacy assembly", exc_info=True)
        system = MAESTRO_SYSTEM
        if memory_recall_context:
            system = memory_recall_context + "\n\n" + system
        if rag_context:
            system = (
                system
                + "\n\nРелевантный контекст из истории переписок:\n"
                + rag_context
            )
        if owner_id is not None:
            try:
                from src.core.adaptive_instructions import format_rules_for_prompt

                rules_hint = await format_rules_for_prompt(owner_id)
                if rules_hint:
                    system += rules_hint
            except Exception:
                pass
        if owner_id is not None:
            try:
                from src.core.adaptive_persona import format_persona_for_prompt

                persona_hint = await format_persona_for_prompt(owner_id)
                if persona_hint:
                    system += persona_hint
            except Exception:
                pass

    try:
        raw = await provider.chat(
            [
                ChatMessage(role="system", content=system),
                ChatMessage(role="user", content=user_msg),
            ],
            heavy=True,
        )
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json|JSON)?\s*\n?", "", raw)
            raw = re.sub(r"\n?\s*```\s*$", "", raw)
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            return json.loads(m.group(0))
        return {
            "understood": raw,
            "plan": [],
            "agents_to_call": [],
            "final_response": raw,
        }
    except Exception:
        logger.exception("Maestro failed")
        return {
            "understood": "не понял",
            "plan": [],
            "agents_to_call": [],
            "final_response": "Извини, я не понял. Повтори пожалуйста.",
        }


# ---- Agent dispatch table ----


def _agent_result_as_text(agent_type: str, result: dict) -> str:
    """Форматирует результат агента для вставки в промпт."""
    if not result.get("success", True):
        err = result.get("error", "неизвестная ошибка")
        return f"[{agent_type}] ❌ Ошибка: {err}"

    data = result.get("data", {})
    if not data:
        return f"[{agent_type}] ✅ Выполнен, но данных нет."

    # Сжимаем большие поля
    lines = [f"[{agent_type}]", "Найдено:"]
    for k, v in data.items():
        s = str(v)
        if len(s) > 400:
            s = s[:400] + "…"
        lines.append(f"  {k}: {s}")
    return "\n".join(lines)


async def _execute_agent(
    provider,
    agent_spec: dict,
    *,
    owner_id: int,
) -> dict:
    """Исполняет одного агента по спецификации из плана maestro."""
    from src.db.session import get_session

    agent_type = agent_spec.get("agent", "")
    query = agent_spec.get("query", "")
    cache = agent_spec.get("cache", True)
    cache_ttl = 300 if cache else 0

    try:
        if agent_type == "search":
            from src.agents.search_agent import resolve as search_resolve
            from src.db.repo import get_or_create_user, list_contacts

            async with get_session() as session:
                owner = await get_or_create_user(session, owner_id)
                contacts = await list_contacts(session, owner)
                contact_dicts = [
                    {"id": c.peer_id, "name": c.display_name, "username": c.username}
                    for c in contacts[:50]
                ]
            data = await search_resolve(provider, query, contact_dicts)
            return {"data": data, "success": True}

        elif agent_type == "memory":
            from src.agents.memory_agent import recall as memory_recall
            from src.db.repo import get_or_create_user, search_memories

            async with get_session() as session:
                owner = await get_or_create_user(session, owner_id)
                facts_obj = await search_memories(session, owner, query)
                facts_list = [m.fact for m in facts_obj] if facts_obj else []
            data = await memory_recall(provider, query, facts_list)
            return {"data": data, "success": True}

        elif agent_type == "urgency":
            from src.core.urgency_classifier import classify_message

            urgency = classify_message(query)
            return {"data": {"urgency": urgency}, "success": True}

        elif agent_type == "commitment":
            from src.agents.commitment_agent import extract as comm_extract

            data = await comm_extract(provider, query)
            return {"data": data, "success": True}

        elif agent_type == "summarizer":
            from src.agents.summarizer_agent import summarize as summ_summarize

            data = await summ_summarize(provider, query)
            return {"data": data, "success": True}

        elif agent_type == "draft":
            from src.agents.draft_agent import draft as draft_agent

            data = await draft_agent(provider, "собеседник", query)
            return {"data": data, "success": True}

        elif agent_type == "digest":
            from src.agents.digest_agent import build_digest as digest_build

            data = await digest_build(provider, [{"text": query}])
            return {"data": data, "success": True}

        else:
            return {
                "data": {},
                "success": False,
                "error": f"Неизвестный агент: {agent_type}",
            }

    except Exception as e:
        logger.exception("Agent %s failed", agent_type)
        return {"data": {}, "success": False, "error": str(e)}


async def _execute_agents_parallel(
    provider, agents_to_call: list, *, owner_id: int
) -> list[dict]:
    """Запускает нескольких агентов параллельно (каждый со своей сессией БД)."""
    if not agents_to_call:
        return []

    tasks = [
        _execute_agent(provider, spec, owner_id=owner_id) for spec in agents_to_call
    ]
    return await asyncio.gather(*tasks, return_exceptions=False)


async def run_pipeline(
    provider,
    user_text: str,
    *,
    owner_id: int,
    history_block: str | None = None,
    memory_context: str | None = None,
    global_style: str | None = None,
    self_profile: str | None = None,
    rag_enabled: bool = True,
) -> dict[str, Any]:
    """Полный пайплайн: Maestro → агенты → финальный ответ.

    Returns:
        dict с ключами:
          - final_response: str (всегда — текст для пользователя)
          - plan: list (план действий)
          - used_agents: list[str] (какие агенты сработали)
          - agent_errors: list[str] (ошибки агентов)
    """
    # --- Загружаем self-profile, если не передан ---
    if self_profile is None:
        try:
            from src.db.models import SelfProfile
            from src.db.repo import get_or_create_user, get_self_profile
            from src.db.session import get_session

            async with get_session() as session:
                owner = await get_or_create_user(session, owner_id)
                profile = await get_self_profile(session, owner)
                if profile:
                    lines = ["ТВОЙ ПРОФИЛЬ (владелец):"]
                    if profile.preferences:
                        lines.append(f"Предпочтения: {profile.preferences}")
                    if profile.goals:
                        lines.append(f"Цели: {profile.goals}")
                    if profile.current_projects:
                        lines.append(f"Проекты: {profile.current_projects}")
                    if profile.decision_style:
                        lines.append(f"Стиль решений: {profile.decision_style}")
                    if profile.communication_preferences:
                        lines.append(
                            f"Коммуникация: {profile.communication_preferences}"
                        )
                    if profile.sleep_pattern:
                        lines.append(f"Сон: {profile.sleep_pattern}")
                    if profile.work_hours:
                        lines.append(f"Рабочие часы: {profile.work_hours}")
                    self_profile = "\n".join(lines)
        except Exception:
            logger.debug("Failed to load self_profile, continuing without")

    # Maestro-only context compression: fast_route never reaches this function.
    try:
        from src.core.context_compressor import compress_maestro_context

        compressed = compress_maestro_context(
            history_block=history_block,
            memory_context=memory_context,
        )
        if compressed.compressed_context and (memory_context or history_block):
            memory_context = compressed.compressed_context
            history_block = None
    except Exception:
        logger.debug("Context compression skipped", exc_info=True)

    # --- Шаг 1: Maestro планирует ---
    plan = await process(
        provider,
        user_text,
        owner_id=owner_id,
        history_block=history_block,
        memory_context=memory_context,
        global_style=global_style,
        self_profile=self_profile,
        rag_enabled=rag_enabled,
    )

    used_agents = []
    agent_errors = []

    # Если Maestro хочет уточнить — показываем вопрос и ждём ответа
    clarification = plan.get("needs_clarification")
    if clarification:
        return {
            "final_response": f"🤔 {clarification}",
            "plan": plan.get("plan", []),
            "used_agents": [],
            "agent_errors": [],
            "is_clarification": True,
        }

    # Если Maestro ответил сам — всё
    if plan.get("final_response"):
        return {
            "final_response": plan["final_response"],
            "plan": plan.get("plan", []),
            "used_agents": [],
            "agent_errors": [],
        }

    # --- Шаг 2: Запустить агентов ---
    agents_to_call = plan.get("agents_to_call", [])
    if not agents_to_call:
        # Нет агентов и нет ответа — показываем understood или clarification
        msg = (
            plan.get("final_response")
            or plan.get("needs_clarification")
            or plan.get("understood", "Не понял. Повтори.")
        )
        if plan.get("needs_clarification"):
            msg = f"🤔 {msg}"
        return {
            "final_response": plan.get("understood", "Не понял. Повтори."),
            "plan": plan.get("plan", []),
            "used_agents": [],
            "agent_errors": [],
        }

    results = await _execute_agents_parallel(
        provider, agents_to_call, owner_id=owner_id
    )

    # Собираем результаты
    agent_texts = []
    for i, r in enumerate(results):
        agent_type = agents_to_call[i].get("agent", "?")
        if r.get("success"):
            used_agents.append(agent_type)
            agent_texts.append(_agent_result_as_text(agent_type, r))
        else:
            err = r.get("error", "неизвестная ошибка")
            agent_errors.append(f"{agent_type}: {err}")
            logger.warning(
                "Agent %s failed: %s — retrying via fallback", agent_type, err
            )

    # --- Шаг 3: Fallback — перезапросить у Maestro с учётом ошибок ---
    if agent_errors and not agent_texts:
        # Ни один агент не сработал — Maestro должен ответить сам
        fallback_prompt = (
            f"Все агенты не справились:\n"
            + "\n".join(agent_errors)
            + f"\n\nОтветь пользователю сам: {user_text}"
        )
        try:
            raw = await provider.chat(
                [
                    ChatMessage(role="system", content=MAESTRO_SYSTEM),
                    ChatMessage(role="user", content=fallback_prompt),
                ],
                heavy=True,
            )
            return {
                "final_response": raw.strip(),
                "plan": plan.get("plan", []),
                "used_agents": [],
                "agent_errors": agent_errors,
            }
        except Exception:
            return {
                "final_response": "Извини, что-то пошло не так. Попробуй ещё раз.",
                "plan": [],
                "used_agents": [],
                "agent_errors": agent_errors,
            }

    # --- Шаг 4: Агенты сработали — просим Maestro сформулировать ответ ---
    if agent_texts:
        combined = "\n\n".join(agent_texts)
        promo = MAESTRO_AFTER_AGENTS.format(agent_results=combined)
        try:
            raw = await provider.chat(
                [
                    ChatMessage(role="system", content=promo),
                    ChatMessage(
                        role="user", content=f"Пользователь сказал: {user_text}"
                    ),
                ],
                heavy=True,
            )
            raw = raw.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json|JSON)?\s*\n?", "", raw)
                raw = re.sub(r"\n?\s*```\s*$", "", raw)
            m = re.search(r"\{[\s\S]*\}", raw)
            if m:
                parsed = json.loads(m.group(0))
                return {
                    "final_response": parsed.get("final_response", raw),
                    "plan": plan.get("plan", []),
                    "used_agents": used_agents,
                    "agent_errors": agent_errors,
                }
            return {
                "final_response": raw,
                "plan": plan.get("plan", []),
                "used_agents": used_agents,
                "agent_errors": agent_errors,
            }
        except Exception:
            # Если LLM не может сформулировать — возвращаем сырые данные агентов
            summary = "\n\n".join(agent_texts)
            return {
                "final_response": f"Вот что я выяснил:\n\n{summary[:1500]}",
                "plan": plan.get("plan", []),
                "used_agents": used_agents,
                "agent_errors": agent_errors,
            }

    # --- Ни один агент не дал результатов ---
    return {
        "final_response": plan.get("final_response")
        or plan.get("understood", "Не получилось выполнить. Попробуй иначе."),
        "plan": plan.get("plan", []),
        "used_agents": used_agents,
        "agent_errors": agent_errors,
    }
