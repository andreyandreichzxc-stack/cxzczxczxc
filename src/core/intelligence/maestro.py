"""Maestro — главный ИИ-координатор. Тяжёлая модель. Планирует и делегирует сабагентам."""

from __future__ import annotations
import asyncio
import importlib

import json
import logging
import re
from typing import Any

from src.core.infra.text_sanitizer import sanitize_html
from src.core.actions.vector_store import get_vector_store
from src.core.intelligence.agent_orchestrator import (
    AgentOrchestrator,
    AGENT_SPECS,
)
from src.db.repo import get_or_create_user, list_contacts, search_memories
from src.db.session import get_session
from src.llm.base import ChatMessage
from src.llm.router import ExhaustedError


logger = logging.getLogger(__name__)

# ── Глобальный оркестратор агентов ──
# Один экземпляр на всё приложение: кеш, health-трекинг, таймауты.
orchestrator = AgentOrchestrator(AGENT_SPECS)

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
- **search** — найди контакт/чат/сообщение по имени или запросу
- **memory** — вспомни/сохрани факты о человеке (предпочтения, прошлые темы)
- **draft** — напиши черновик ответа
- **summarizer** — сводка переписки, «где остановились»
- **digest** — дайджест входящих
- **commitment** — извлеки обещания, дедлайны
- **urgency** — насколько срочное сообщение

## Все доступные действия (intents) — полный каталог

### 📝 Сообщения и переписка
- **send_message** — отправить сообщение контакту
- **draft_reply** — черновик ответа для контакта
- **summarize_chat** — саммари переписки с контактом
- **catchup** — восстановить контекст переписки (где остановились)
- **tasks_for_chat** — извлечь задачи/обещания из переписки
- **add_reminders_from_chat** — извлечь напоминания из чата
- **extract_memories_from_chat** — извлечь факты из переписки

### 🧠 Память
- **store_memory** — сохранить факт о владельце или контакте
- **list_memories** — показать сохранённые факты
- **check_memories** — проверить старые/негативные факты
- **forget_memory** — удалить факты

### 🔍 Поиск и информация
- **search** — поиск по сообщениям
- **find_in_chats** — найти чат по теме
- **show_inbox** — показать входящие сообщения
- **show_digest** — сводка непрочитанного / утренний дайджест
- **show_today** — сводка событий за сегодня
- **show_threads** — активные диалоги
- **show_trajectory** — история действий пользователя
- **show_skills** — список возможностей бота
- **show_profile** — профиль пользователя
- **show_self** — профиль владельца
- **show_style** — стиль общения
- **list_keys** — список API-ключей

### ⏰ Напоминания
- **add_reminder** — поставить напоминание
- **remove_reminder** — снять напоминание
- **list_todos** — открытые обещания/напоминания

### 📰 Новости
- **news_digest** — новостной дайджест по теме
- **add_news_topic** — добавить тему новостей
- **remove_news_topic** — удалить тему новостей

### ⚙️ Настройки
- **set_setting** — изменить любую настройку
- **change_auto_mode** — режим авто-ответа
- **set_quiet_hours** — тихие часы
- **add_api_key** — добавить API-ключ
- **remove_api_key** — удалить слот ключа
- **toggle_api_key** — включить/выключить ключ

### 📊 Аналитика
- **full_analysis** — полный анализ переписок
- **index_chats** — переиндексировать чаты

### 💬 Базовые
- **chat** — просто ответить (болтовня, совет, вопрос)
- **clarify** — уточнить у пользователя
- **multi** — выполнить несколько действий одновременно

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
- **НИКОГДА не переспрашивай уточнения, если информация уже есть в контексте диалога.**
- **Если пользователь говорит «контакты», «мои контакты», «список контактов» — сразу показывай список, не спрашивая.**
- **Действуй, а не спрашивай.** Только если ДЕЙСТВИТЕЛЬНО непонятно (2+ равновероятных варианта) — уточни ОДИН раз.
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
        _owner_db_id = None
        try:
            async with get_session() as session:
                owner_db = await get_or_create_user(session, owner_id)
                _owner_db_id = owner_db.id if owner_db else None
            if _owner_db_id is not None:
                query_vec = await provider.embed(user_text)
                hits = await get_vector_store().search(
                    user_id=_owner_db_id, embedding=query_vec, limit=5
                )
            else:
                hits = []
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
            from src.core.memory.memory_recall import recall, format_recall_for_prompt

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
        from src.core.intelligence.prompt_assembler import (
            AssemblyContext,
            prompt_assembler,
        )

        # Собираем persona блок
        persona_block = ""
        if owner_id is not None:
            try:
                from src.core.intelligence.adaptive_persona import (
                    format_persona_for_prompt,
                )

                persona_block = await format_persona_for_prompt(owner_id) or ""
            except Exception:
                pass

        # Собираем style‑match блок (динамический анализ стиля пользователя)
        style_match_block = ""
        if owner_id is not None:
            try:
                from src.core.intelligence.style_matcher import (
                    get_or_update_style_profile,
                )

                style_match_block = await get_or_update_style_profile(owner_id) or ""
            except Exception:
                logger.debug("Style matcher skipped", exc_info=True)

        # Собираем confirmed rules
        confirmed_rules = []
        if owner_id is not None:
            try:
                from src.core.intelligence.adaptive_instructions import get_active_rules

                confirmed_rules = await get_active_rules(owner_id)
            except Exception:
                pass

        # Load anti-AI setting from user settings
        anti_ai = False
        if owner_id is not None:
            try:
                async with get_session() as _s:
                    _owner = await get_or_create_user(_s, owner_id)
                    anti_ai = _owner.settings.anti_ai_enabled
            except Exception:
                pass

        ctx = AssemblyContext(
            target="maestro",
            user_id=owner_id or 0,
            user_message=user_text,
            memory_context=memory_recall_context,
            rag_context=rag_context,
            persona_block=persona_block,
            style_match_block=style_match_block,
            confirmed_rules=confirmed_rules,
            anti_ai=anti_ai,
            history_block=history_block or "",
        )
        if owner_id is not None:
            try:
                from src.core.intelligence.skills import build_skill_index

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
                from src.core.intelligence.adaptive_instructions import (
                    format_rules_for_prompt,
                )

                rules_hint = await format_rules_for_prompt(owner_id)
                if rules_hint:
                    system += rules_hint
            except Exception:
                pass
        if owner_id is not None:
            try:
                from src.core.intelligence.adaptive_persona import (
                    format_persona_for_prompt,
                )

                persona_hint = await format_persona_for_prompt(owner_id)
                if persona_hint:
                    system += persona_hint
            except Exception:
                pass

    try:
        raw = await asyncio.wait_for(
            provider.chat(
                [
                    ChatMessage(role="system", content=system),
                    ChatMessage(role="user", content=user_msg),
                ],
                heavy=True,
            ),
            timeout=60.0,
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
    except ExhaustedError:
        logger.warning("Maestro ExhaustedError during process")
        return {
            "understood": "нет ключей",
            "plan": [],
            "agents_to_call": [],
            "final_response": "🔑 Все API-ключи исчерпаны. Добавь новые через /keys add ...",
        }
    except asyncio.TimeoutError:
        logger.warning("Maestro TimeoutError during process")
        return {
            "understood": "таймаут",
            "plan": [],
            "agents_to_call": [],
            "final_response": "⏱️ Ответ занял слишком много времени. Попробуй короче.",
        }
    except Exception as e:
        if "context_length" in str(e).lower() or "token" in str(e).lower():
            logger.warning("Maestro context overflow: %s", e)
            return {
                "understood": "контекст переполнен",
                "plan": [],
                "agents_to_call": [],
                "final_response": "📏 Контекст переполнен. Упрости запрос или уменьши историю.",
            }
        if "rate" in str(e).lower():
            logger.warning("Maestro rate limit: %s", e)
            return {
                "understood": "лимит",
                "plan": [],
                "agents_to_call": [],
                "final_response": "🚦 Превышен лимит запросов. Подожди минуту.",
            }
        logger.exception("Maestro failed")
        return {
            "understood": "не понял",
            "plan": [],
            "agents_to_call": [],
            "final_response": "Извини, я не понял. Повтори пожалуйста.",
        }


# ---- Agent dispatch table ----

AGENT_REGISTRY: dict[str, tuple[str, str]] = {
    "search": ("src.agents.search_agent", "resolve"),
    "memory": ("src.agents.memory_agent", "recall"),
    "urgency": ("src.core.contacts.urgency_classifier", "classify_message"),
    "commitment": ("src.agents.commitment_agent", "extract"),
    "summarizer": ("src.agents.summarizer_agent", "summarize"),
    "draft": ("src.agents.draft_agent", "draft"),
    "digest": ("src.agents.digest_agent", "build_digest"),
    "skill_creator": ("src.agents.skill_creator_agent", "propose"),
}


async def _invoke_search(func, provider, query, owner_id, **kwargs):
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        contacts = await list_contacts(session, owner)
        contact_dicts = [
            {"id": c.peer_id, "name": c.display_name, "username": c.username}
            for c in contacts[:50]
        ]
    data = await func(provider, query, contact_dicts)
    return {"data": data, "success": True}


async def _invoke_memory(func, provider, query, owner_id, **kwargs):
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        facts_obj = await search_memories(session, owner, query)
        facts_list = [m.fact for m in facts_obj] if facts_obj else []
    data = await func(provider, query, facts_list)
    return {"data": data, "success": True}


async def _invoke_urgency(func, _provider, query, _owner_id, **kwargs):
    urgency = func(query)
    return {"data": {"urgency": urgency}, "success": True}


async def _invoke_commitment(func, provider, query, _owner_id, **kwargs):
    data = await func(provider, query)
    return {"data": data, "success": True}


async def _invoke_summarizer(func, provider, query, _owner_id, **kwargs):
    data = await func(provider, query)
    return {"data": data, "success": True}


async def _invoke_draft(func, provider, query, _owner_id, **kwargs):
    agent_spec = kwargs.get("agent_spec", {})
    contact_name = (
        agent_spec.get("contact_name") or agent_spec.get("sender_name") or "собеседник"
    )
    data = await func(provider, contact_name, query)
    return {"data": data, "success": True}


async def _invoke_digest(func, provider, query, _owner_id, **kwargs):
    data = await func(provider, [{"text": query}])
    return {"data": data, "success": True}


async def _invoke_skill_creator(func, provider, query, owner_id, **kwargs):
    """Вызывает skill_creator агент: собирает последние сообщения и анализирует."""
    async with get_session() as session:
        from src.db.repo import fetch_my_messages_global

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
    data = await func(provider, recent_messages)
    return {"data": data, "success": True}


_AGENT_INVOKERS: dict[str, Any] = {
    "search": _invoke_search,
    "memory": _invoke_memory,
    "urgency": _invoke_urgency,
    "commitment": _invoke_commitment,
    "summarizer": _invoke_summarizer,
    "draft": _invoke_draft,
    "digest": _invoke_digest,
    "skill_creator": _invoke_skill_creator,
}


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
    agent_type = agent_spec.get("agent", "")
    query = agent_spec.get("query", "")

    agent_info = AGENT_REGISTRY.get(agent_type)
    if agent_info is None:
        logger.warning("Unknown agent type: %s", agent_type)
        return {
            "agent": agent_type,
            "data": {},
            "success": False,
            "error": "Неизвестный агент: " + agent_type,
        }

    invoker = _AGENT_INVOKERS.get(agent_type)
    if invoker is None:
        logger.error("No invoker registered for agent: %s", agent_type)
        return {
            "agent": agent_type,
            "data": {},
            "success": False,
            "error": "Нет обработчика для агента: " + agent_type,
        }

    module_path, func_name = agent_info
    try:
        module = importlib.import_module(module_path)
        agent_func = getattr(module, func_name)
        result = await invoker(
            agent_func, provider, query, owner_id, agent_spec=agent_spec
        )
        result["agent"] = agent_type
        return result
    except Exception as e:
        logger.exception("Agent %s failed", agent_type)
        return {"agent": agent_type, "data": {}, "success": False, "error": str(e)}


async def _execute_agents_parallel(
    provider, agents_to_call: list, *, owner_id: int
) -> list[dict]:
    """Запускает нескольких агентов параллельно (каждый со своей сессией БД)."""
    if not agents_to_call:
        return []

    tasks = [
        _execute_agent(provider, spec, owner_id=owner_id) for spec in agents_to_call
    ]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    results: list[dict] = []
    for i, r in enumerate(raw_results):
        if isinstance(r, Exception):
            agent_name = agents_to_call[i].get("agent", "?")
            logger.error("Agent %s failed with exception: %s", agent_name, r)
            results.append(
                {
                    "agent": agent_name,
                    "data": {},
                    "success": False,
                    "error": str(r),
                }
            )
        else:
            results.append(r)
    return results


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
            from src.core.intelligence.prompt_assembler import (
                assemble_self_profile_prompt,
            )

            self_profile = await assemble_self_profile_prompt(owner_id)
        except Exception:
            logger.debug("Failed to load self_profile, continuing without")

    # Maestro-only context compression: fast_route never reaches this function.
    try:
        from src.core.intelligence.context_compressor import compress_maestro_context

        # Build message history from string context blocks
        history: list[dict] = []
        if memory_context:
            history.append({"role": "system", "content": memory_context})
        if history_block:
            history.append({"role": "system", "content": history_block})

        if history:
            context_text, _ = await compress_maestro_context(
                history=history,
                owner_id=owner_id,
            )
            if context_text:
                memory_context = context_text
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
            "final_response": sanitize_html(f"🤔 {clarification}"),
            "plan": plan.get("plan", []),
            "used_agents": [],
            "agent_errors": [],
            "is_clarification": True,
        }

    # Если Maestro ответил сам и агенты не нужны — возвращаем сразу
    agents_to_call = plan.get("agents_to_call", [])
    if plan.get("final_response") and not agents_to_call:
        return {
            "final_response": sanitize_html(plan["final_response"]),
            "plan": plan.get("plan", []),
            "used_agents": [],
            "agent_errors": [],
        }

    # --- Шаг 2: Запустить агентов ---
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
            "final_response": sanitize_html(
                plan.get("understood", "Не понял. Повтори.")
            ),
            "plan": plan.get("plan", []),
            "used_agents": [],
            "agent_errors": [],
        }

    # --- Шаг 2: Запустить агентов через оркестратор ---
    # Оркестратор обеспечивает: per-agent timeout, кеш, health-check,
    # cooldown для repeat-фейлов, partial results (один упал — остальные живы).
    results, orch_errors = await orchestrator.execute(
        agents_to_call, provider, owner_id
    )

    # Собираем результаты
    agent_texts = []
    for r in results:
        agent_type = r.get("agent", "?")
        if r.get("success"):
            used_agents.append(agent_type)
            agent_texts.append(_agent_result_as_text(agent_type, r))
        else:
            err = r.get("error", "неизвестная ошибка")
            agent_errors.append(f"{agent_type}: {err}")
            logger.warning(
                "Agent %s failed: %s — retrying via fallback", agent_type, err
            )

    # Ошибки оркестрации (cooldown, timeout) — тоже в agent_errors
    agent_errors.extend(orch_errors)

    # --- Шаг 3: Fallback — перезапросить у Maestro с учётом ошибок ---
    if agent_errors and not agent_texts:
        # Ни один агент не сработал — Maestro должен ответить сам
        fallback_prompt = (
            "Все агенты не справились:\n"
            + "\n".join(agent_errors)
            + f"\n\nОтветь пользователю сам: {user_text}"
        )
        try:
            raw = await asyncio.wait_for(
                provider.chat(
                    [
                        ChatMessage(role="system", content=MAESTRO_SYSTEM),
                        ChatMessage(role="user", content=fallback_prompt),
                    ],
                    heavy=True,
                ),
                timeout=60.0,
            )
            return {
                "final_response": sanitize_html(raw.strip()),
                "plan": plan.get("plan", []),
                "used_agents": [],
                "agent_errors": agent_errors,
            }
        except ExhaustedError:
            logger.warning("maestro fallback_request ExhaustedError")
            return {
                "final_response": sanitize_html(
                    "🔑 Все API-ключи исчерпаны. Добавь новые через /keys add ..."
                ),
                "plan": [],
                "used_agents": [],
                "agent_errors": agent_errors,
            }
        except asyncio.TimeoutError:
            logger.warning("maestro fallback_request TimeoutError")
            return {
                "final_response": sanitize_html(
                    "⏱️ Ответ занял слишком много времени. Попробуй короче."
                ),
                "plan": [],
                "used_agents": [],
                "agent_errors": agent_errors,
            }
        except Exception as e:
            if "context_length" in str(e).lower() or "token" in str(e).lower():
                logger.warning("maestro fallback_request context overflow: %s", e)
                return {
                    "final_response": sanitize_html(
                        "📏 Контекст переполнен. Упрости запрос или уменьши историю."
                    ),
                    "plan": [],
                    "used_agents": [],
                    "agent_errors": agent_errors,
                }
            if "rate" in str(e).lower():
                logger.warning("maestro fallback_request rate limit: %s", e)
                return {
                    "final_response": sanitize_html(
                        "🚦 Превышен лимит запросов. Подожди минуту."
                    ),
                    "plan": [],
                    "used_agents": [],
                    "agent_errors": agent_errors,
                }
            logger.exception("maestro fallback_request failed")
            return {
                "final_response": sanitize_html(
                    plan.get("final_response")
                    or "Извини, что-то пошло не так. Попробуй ещё раз."
                ),
                "plan": [],
                "used_agents": [],
                "agent_errors": agent_errors,
            }

    # --- Шаг 4: Агенты сработали — просим Maestro сформулировать ответ ---
    if agent_texts:
        combined = "\n\n".join(agent_texts)
        promo = MAESTRO_AFTER_AGENTS.format(agent_results=combined)
        try:
            raw = await asyncio.wait_for(
                provider.chat(
                    [
                        ChatMessage(role="system", content=promo),
                        ChatMessage(
                            role="user", content=f"Пользователь сказал: {user_text}"
                        ),
                    ],
                    heavy=True,
                ),
                timeout=60.0,
            )
            raw = raw.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json|JSON)?\s*\n?", "", raw)
                raw = re.sub(r"\n?\s*```\s*$", "", raw)
            m = re.search(r"\{[\s\S]*\}", raw)
            if m:
                parsed = json.loads(m.group(0))
                return {
                    "final_response": sanitize_html(parsed.get("final_response", raw)),
                    "plan": plan.get("plan", []),
                    "used_agents": used_agents,
                    "agent_errors": agent_errors,
                }
            return {
                "final_response": sanitize_html(raw),
                "plan": plan.get("plan", []),
                "used_agents": used_agents,
                "agent_errors": agent_errors,
            }
        except Exception:
            logger.exception("maestro agent synthesis failed")
            # Если LLM не может сформулировать — возвращаем сырые данные агентов
            summary = "\n\n".join(agent_texts)
            return {
                "final_response": sanitize_html(
                    f"Вот что я выяснил:\n\n{summary[:1500]}"
                ),
                "plan": plan.get("plan", []),
                "used_agents": used_agents,
                "agent_errors": agent_errors,
            }

    # --- Ни один агент не дал результатов ---
    return {
        "final_response": sanitize_html(
            plan.get("final_response")
            or plan.get("understood", "Не получилось выполнить. Попробуй иначе.")
        ),
        "plan": plan.get("plan", []),
        "used_agents": used_agents,
        "agent_errors": agent_errors,
    }
