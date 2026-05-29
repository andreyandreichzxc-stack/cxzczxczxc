"""Maestro — главный ИИ-координатор. Тяжёлая модель. Планирует и делегирует сабагентам."""

from __future__ import annotations
import asyncio
import importlib

import json
import logging
import re
from typing import Any

from src.config import settings
from src.core.infra.key_guard import safe_str
from src.core.infra.text_sanitizer import sanitize_html
from src.core.actions.vector_store import get_vector_store
from src.core.intelligence.agent_orchestrator import (
    AgentOrchestrator,
    AGENT_SPECS,
)
from src.db.repo import get_or_create_user, list_contacts, search_memories
from src.db.session import get_session
from src.llm.base import ChatMessage, TaskType
from src.llm.router import ExhaustedError

from src.core.actions import register_builtin_tools
from src.core.actions.tool_registry import tool_registry
from src.core.intelligence.guardrails import evaluate as guardrail_evaluate

logger = logging.getLogger(__name__)

# ── Fallback chains для инструментов ──
# При ошибке выполнения тула пробуем альтернативы по порядку.
# admit_ignorance — всегда последний рубеж: признаём неспособность и предлагаем поиск.
_TOOL_FALLBACKS = {
    "web_search": ["mcp_web", "admit_ignorance"],
    "analyze_image": ["admit_ignorance"],
    "code_exec": ["admit_ignorance"],
    "mcp_youtube": ["admit_ignorance"],
}

# ── Максимальное число итераций в tool‑loop ──
MAX_TOOL_ITERATIONS = settings.max_tool_iterations

# ── Глобальный оркестратор агентов ──
# Один экземпляр на всё приложение: кеш, health-трекинг, таймауты.
orchestrator = AgentOrchestrator(AGENT_SPECS)

# ── Fallback подсказки, когда бот не понял запрос ──
FALLBACK_HINTS = (
    "🤔 Я не совсем понял. Попробуй одну из команд:\n\n"
    "👤 /contact Имя — что я знаю о человеке\n"
    "📅 /timeline тема — где обсуждали X\n"
    "📝 /send Имя текст — написать человеку\n"
    "🔍 /search запрос — найти в чатах\n"
    "📋 /todos — твои обещания\n"
    "📰 /news тема — дайджест каналов\n\n"
    "Или просто напиши обычным языком — я попробую понять."
)


def _extract_json_object(raw: str) -> dict | None:
    """Return the first valid JSON object embedded in model output."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw):
        try:
            value, _end = decoder.raw_decode(raw[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


from src.core.intelligence.soul_blocks import MAESTRO_SYSTEM_FULL as MAESTRO_SYSTEM

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
    contact_id: int | None = None,
    userbot_manager: Any | None = None,
) -> dict[str, Any]:
    """Главная точка входа. Maestro понимает пользователя и составляет план."""
    register_builtin_tools()

    # Override provider model if maestro_model is configured
    maestro_model = getattr(settings, "maestro_model", None)
    if maestro_model and not getattr(provider, "_model", None):
        try:
            provider._model = maestro_model  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            pass

    ctx_parts = []
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

    # --- Modular prompt assembly (Block 4) ---
    ctx = None
    _used_skills_meta: list[dict] = []
    frozen_snapshot_injected = False
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
                logger.debug("Failed to format persona for prompt", exc_info=True)

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
                logger.debug("Failed to load active rules", exc_info=True)

        # Load anti-AI setting from user settings
        anti_ai = False
        if owner_id is not None:
            try:
                async with get_session() as _s:
                    _owner = await get_or_create_user(_s, owner_id)
                    anti_ai = _owner.settings.anti_ai_enabled
            except Exception:
                logger.debug("Failed to load anti_ai setting", exc_info=True)

        # Pre-load recent corrections for context injection
        correction_context = ""
        if owner_id is not None:
            try:
                from src.core.intelligence.correction_learner import (
                    get_recent_corrections,
                )

                corrections = await get_recent_corrections(owner_id, limit=3)
                if corrections:
                    correction_context = "; ".join(
                        f'"{c["original"][:80]}" → "{c["corrected"][:80]}"'
                        for c in corrections
                    )
            except Exception:
                logger.debug("Failed to load correction context", exc_info=True)

        # --- Voice transcription metadata ---
        _transcription_meta = None
        if owner_id is not None:
            try:
                from src.core.memory.conversation_context import (
                    get_and_clear_transcription_meta,
                )

                _transcription_meta = await get_and_clear_transcription_meta(owner_id)
            except Exception:
                logger.debug("Failed to load transcription_meta", exc_info=True)

        ctx = AssemblyContext(
            target="maestro",
            user_id=owner_id or 0,
            user_message=user_text,
            rag_context=rag_context,
            persona_block=persona_block,
            style_match_block=style_match_block,
            confirmed_rules=confirmed_rules,
            anti_ai=anti_ai,
            history_block=history_block or "",
            memory_context=memory_context or "",
            self_profile=self_profile or "",
            correction_context=correction_context,
            transcription_meta=_transcription_meta,
        )
        _used_skills_meta: list[dict] = []
        if owner_id is not None:
            try:
                from src.core.intelligence.skills import build_skill_index

                skill_str, skill_meta = await build_skill_index(
                    owner_id, user_text, "maestro"
                )
                ctx.skill_index = skill_str
                _used_skills_meta = skill_meta
            except Exception:
                logger.debug("Failed to build skill index", exc_info=True)

        # --- Frozen memory snapshot: top-3 facts pre-loaded ---
        frozen_snapshot_injected = False
        if owner_id is not None:
            try:
                from src.core.memory.memory_recall import recall

                _recall_result = await recall(
                    telegram_id=owner_id,
                    query=user_text,
                    limit=3,
                    include_deep=False,
                    mode="normal",
                )
                if _recall_result.facts:
                    _lines = [
                        "[ПАМЯТЬ] Ниже факты о пользователе и его контактах. "
                        "Используй их ЕСТЕСТВЕННО в ответе — не перечисляй списком, "
                        "не говори «я помню» или «по моим данным». "
                        "Вплетай в речь как само собой разумеющееся."
                    ]
                    for _f in _recall_result.facts:
                        _lines.append(f"[{_f.reason}] {_f.fact}")
                    ctx.frozen_snapshot = "\n".join(_lines)
                    frozen_snapshot_injected = True

                    # Also update the frozen_provider so ContextEngine can serve it
                    try:
                        from src.core.context.providers.frozen_provider import (
                            frozen_provider,
                        )

                        await frozen_provider.set_frozen(
                            owner_id,
                            [
                                {"fact": f"[{_f.reason}] {_f.fact}"}
                                for _f in _recall_result.facts
                            ],
                        )
                    except Exception:
                        logger.debug("Failed to set frozen provider", exc_info=True)
            except Exception:
                logger.debug("Frozen snapshot recall failed, skipping", exc_info=True)

        context_chunks = []

        # --- ContextEngine: pluggable context providers ---
        # Keep the legacy recall paths above for compatibility, but also let
        # registered providers contribute a compact unified context block.
        if owner_id is not None:
            try:
                from src.core.context.engine import engine as context_engine

                context_chunks = await context_engine.gather(
                    user_text,
                    telegram_id=owner_id,
                    contact_id=contact_id,
                    limit=6,
                )
            except Exception:
                logger.debug("ContextEngine gather failed, skipping", exc_info=True)

        from src.core.context.runtime_bundle import build_runtime_context

        runtime_context = build_runtime_context(
            memory_context=ctx.memory_context,
            self_profile=ctx.self_profile,
            chunks=context_chunks[:10],
        )
        ctx.memory_context = runtime_context.memory_context
        ctx.self_profile = runtime_context.self_profile

        # --- Contact-specific rules (pre-load for prompt injection) ---
        if contact_id and contact_id > 0 and owner_id is not None:
            try:
                from src.core.contacts.contact_rules import get_contact_rules_block

                _block = await get_contact_rules_block(owner_id, contact_id)
                if _block:
                    ctx.contact_rules_block = _block
            except Exception:
                logger.debug("Failed to load contact rules block", exc_info=True)

        # --- DSM: cross-session project memory (pre-load for prompt injection) ---
        try:
            from src.core.intelligence.dsm import dsm_get_recent

            dsm_entries = await dsm_get_recent(limit=5)
            if dsm_entries:
                ctx.dsm_context = "[ПРОЕКТНАЯ ПАМЯТЬ]\n" + "\n".join(
                    f"- [{r['tags'] or 'общее'}] {r['content'][:200]}"
                    for r in dsm_entries
                )
        except Exception:
            logger.debug("Failed to load DSM context", exc_info=True)

        # --- Contact graph: build cross-contact relationship graph ---
        if owner_id is not None:
            try:
                from src.core.memory.memory_neighbors import get_contact_graph

                graph = await get_contact_graph(owner_id, limit=20)
                if graph.get("edges"):
                    lines = []
                    for edge in graph["edges"]:
                        lines.append(
                            f"{edge['from']} ↔ {edge['to']} ({edge['relation']})"
                        )
                    ctx.contact_graph = "\n".join(lines)
            except Exception:
                logger.debug("Failed to build contact graph", exc_info=True)

        system = prompt_assembler.assemble(ctx)
    except Exception:
        # Fallback: старая сборка (обратная совместимость)
        logger.debug("Prompt assembler failed, using legacy assembly", exc_info=True)
        system = MAESTRO_SYSTEM
        if rag_context:
            system = (
                system
                + "\n\nРелевантный контекст из истории переписок:\n"
                + rag_context
            )
        if memory_context:
            system = system + "\n\nФакты из памяти:\n" + memory_context
        if owner_id is not None:
            try:
                from src.core.intelligence.adaptive_instructions import (
                    format_rules_for_prompt,
                )

                rules_hint = await format_rules_for_prompt(owner_id)
                if rules_hint:
                    system += rules_hint
            except Exception:
                logger.debug(
                    "Failed to format rules for prompt (fallback)", exc_info=True
                )
        if owner_id is not None:
            try:
                from src.core.intelligence.adaptive_persona import (
                    format_persona_for_prompt,
                )

                persona_hint = await format_persona_for_prompt(owner_id)
                if persona_hint:
                    system += persona_hint
            except Exception:
                logger.debug(
                    "Failed to format persona for prompt (fallback)", exc_info=True
                )

    # ── Append available tools to system prompt ──
    tools_section = (
        "\n\n## Доступные инструменты\n"
        "### Для вызова инструмента используй JSON формата "
        '`{"tool": "имя", "params": {...}}`.\n'
        "### Для обычного ответа используй "
        '`{"final_response": "твой ответ"}`.\n\n'
        + tool_registry.format_tools_with_schemas()
        + "\n\n"
        "### Факт-чекинг\n"
        "Если тебя спрашивают о факте, который мог измениться"
        " (кто президент, курс валют, погода, новости, население,"
        " дата события, законы, технологии):\n"
        '1. Вызови `mcp_web` c `action="search"` — найди актуальные источники.\n'
        '2. Вызови `mcp_web` c `action="fetch"` — получи детали с лучшего результата.\n'
        "3. Ответь на основе полученных данных, укажи источник.\n"
        "Не полагайся на свои внутренние знания для вопросов,"
        " ответ на который мог устареть."
    )
    if frozen_snapshot_injected:
        tools_section += (
            "\n\nВ системном промпте уже есть топ-3 факта из памяти. "
            "Если их недостаточно — используй инструмент recall_memory."
        )
    system += tools_section

    # ── Inject skill documentation ──
    from src.core.intelligence.skill_docs import list_skill_docs

    skill_docs = list_skill_docs()
    if skill_docs:
        system += "\n\n## Доступные навыки\n"
        for doc in skill_docs:
            lines = doc["content"].split("\n")
            purpose_line = lines[3] if len(lines) > 3 else ""
            system += f"- **{doc['name']}**: {purpose_line}\n"

    # ── Tool‑calling loop ──
    messages = [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=user_msg),
    ]
    trace: dict[str, Any] = {
        "route": "maestro",
        "context_sources": [],
        "memory_facts_count": 0,
        "tools_proposed": [],
        "tools_executed": [],
        "tools_blocked": [],
        "guardrail_decision": {},
    }
    if ctx is not None and ctx.memory_context:
        trace["memory_facts_count"] = sum(
            1
            for line in ctx.memory_context.splitlines()
            if line.strip().startswith(("-", "*", "•", "["))
        )
        for marker in ("recall_context", "context_engine", "self_profile"):
            if marker in ctx.memory_context:
                trace["context_sources"].append(marker)

    for iteration in range(MAX_TOOL_ITERATIONS):
        try:
            raw = await asyncio.wait_for(
                provider.chat(messages, task_type=TaskType.MAESTRO),
                timeout=60.0,
            )
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
            if (
                "context_length" in safe_str(e).lower()
                or "token" in safe_str(e).lower()
            ):
                logger.warning("Maestro context overflow: %s", e)
                return {
                    "understood": "контекст переполнен",
                    "plan": [],
                    "agents_to_call": [],
                    "final_response": "📏 Контекст переполнен. Упрости запрос или уменьши историю.",
                }
            if "rate" in safe_str(e).lower():
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
                "final_response": FALLBACK_HINTS,
            }

        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json|JSON)?\s*\n?", "", raw)
            raw = re.sub(r"\n?\s*```\s*$", "", raw)
        parsed = _extract_json_object(raw)
        if parsed is None:
            # Non‑JSON → treat as final_response text
            return {
                "understood": raw,
                "plan": [],
                "agents_to_call": [],
                "final_response": raw,
            }

        # ── Confidence & admit_ignorance check ──
        try:
            confidence = float(parsed.get("confidence", 0.8))
        except (ValueError, TypeError):
            confidence = 0.8
        confidence = max(0.0, min(1.0, confidence))
        intent = parsed.get("intent", "")

        # Если низкий confidence — переспрашиваем или признаёмся
        if confidence < 0.5 and intent != "clarify" and intent != "admit_ignorance":
            intent = "admit_ignorance"
            parsed["intent"] = intent

        # ── Tool call? ──
        if (
            isinstance(parsed, dict)
            and "tool" in parsed
            and isinstance(parsed["tool"], str)
            and "params" in parsed
            and isinstance(parsed["params"], dict)
        ):
            tool_name = parsed["tool"]
            tool_params = parsed["params"]
            trace["tools_proposed"].append(tool_name)

            # Guardrails evaluate
            gr = guardrail_evaluate(tool_name, tool_params)
            trace["guardrail_decision"] = {
                "tool": tool_name,
                "risk": gr.risk.value,
                "needs_confirm": gr.needs_confirm,
            }
            if gr.needs_confirm:
                trace["tools_blocked"].append(tool_name)
                return {
                    "understood": f"tool_confirmation: {tool_name}",
                    "plan": [],
                    "agents_to_call": [],
                    "final_response": gr.confirm_message,
                    "needs_clarification": None,
                    "confirmation_needed": True,
                    "confirm_message": gr.confirm_message,
                    "tool": tool_name,
                    "tool_params": gr.sanitized_params,
                    "trace": trace,
                }

            # Execute tool with runtime dependencies
            # Open a DB session and resolve the User ORM for the duration
            # of the tool call.  The session stays alive until the tool
            # returns, then commits/rolls back automatically.
            runtime_kwargs: dict[str, Any] = {"provider": provider}
            if userbot_manager is not None:
                runtime_kwargs["userbot_manager"] = userbot_manager

            tool_result = None

            if owner_id is not None:
                async with get_session() as session:
                    owner = await get_or_create_user(session, owner_id)
                    runtime_kwargs["session"] = session
                    runtime_kwargs["user"] = owner
                    if userbot_manager is not None:
                        client = userbot_manager.get_client(owner_id)
                        if client is not None:
                            runtime_kwargs["client"] = client
                    # ── Tool execution with fallback chains (Plan B) ──
                    try:
                        tool_result = await tool_registry.execute(
                            tool_name,
                            _confirmed=False,
                            **gr.sanitized_params,
                            **runtime_kwargs,
                        )
                    except Exception as e:
                        tool_result = {"error": str(e)}
            else:
                # ── Tool execution with fallback chains (Plan B) ──
                try:
                    tool_result = await tool_registry.execute(
                        tool_name,
                        _confirmed=False,
                        **gr.sanitized_params,
                        **runtime_kwargs,
                    )
                except Exception as e:
                    tool_result = {"error": str(e)}

            # Fallback chains — если инструмент вернул ошибку, пробуем альтернативы
            if tool_result and isinstance(tool_result, dict) and "error" in tool_result:
                fallbacks = _TOOL_FALLBACKS.get(tool_name, ["admit_ignorance"])
                logger.warning(
                    "Tool '%s' failed: %s. Trying fallbacks: %s",
                    tool_name,
                    tool_result.get("error", "unknown"),
                    fallbacks,
                )
                for fb in fallbacks:
                    if fb == "admit_ignorance":
                        intent = "admit_ignorance"
                        parsed["intent"] = intent
                        tool_result = None
                        break
                    # Пробуем альтернативный инструмент
                    try:
                        fb_result = await tool_registry.execute(
                            fb,
                            _confirmed=False,
                            **gr.sanitized_params,
                            **runtime_kwargs,
                        )
                        if isinstance(fb_result, dict) and "error" in fb_result:
                            continue  # этот fallback тоже с ошибкой, пробуем следующий
                        tool_result = fb_result
                        tool_name = fb  # запомнили что сработал fallback
                        logger.info(
                            "Fallback '%s' succeeded for '%s'", fb, parsed["tool"]
                        )
                        break
                    except Exception:
                        continue
                else:
                    # Все fallback'и исчерпаны — последний рубеж: admit_ignorance
                    intent = "admit_ignorance"
                    parsed["intent"] = intent
                    tool_result = None
                    logger.warning(
                        "All fallbacks exhausted for '%s', falling back to admit_ignorance",
                        parsed["tool"],
                    )

            # Если fallback привёл к admit_ignorance — обработаем ниже
            if intent == "admit_ignorance":
                pass  # fall through к admit_ignorance handler
            elif tool_result is not None:
                # Feed result back to LLM
                trace["tools_executed"].append(tool_name)
                result_str = json.dumps(tool_result, ensure_ascii=False, default=str)
                if len(result_str) > 4000:
                    result_str = result_str[:4000] + "…"
                messages.append(
                    ChatMessage(
                        role="system",
                        content=f"Tool result ({tool_name}): {result_str}",
                    )
                )
                # Continue loop for next LLM response
                continue
            else:
                # No result and not admit_ignorance — skip, continue loop
                continue

        # ── admit_ignorance handler ──
        if intent == "admit_ignorance":
            # Сначала пробуем веб-поиск — может, ответ уже есть в интернете
            try:
                from src.core.actions.mcp_web_search import web_search

                search_result = await web_search(query=user_text[:300], limit=3)
                if search_result.get("ok") and search_result.get("results"):
                    # Нашли результаты — просим модель ответить на основе поиска
                    snippets = "\n".join(
                        f"- {r['title']}: {r['snippet']}"
                        for r in search_result["results"]
                    )
                    search_prompt = (
                        f"Пользователь спросил: {user_text[:500]}\n\n"
                        f"Я нашёл в интернете:\n{snippets}\n\n"
                        f"Дай краткий ответ на основе этих данных. Если данные неполные — так и скажи."
                    )
                    resp = await provider.chat(
                        [ChatMessage(role="user", content=search_prompt)],
                        task_type=TaskType.DEFAULT,
                    )
                    return {
                        "understood": "ответ через веб-поиск",
                        "plan": [],
                        "agents_to_call": [],
                        "final_response": resp,
                        "needs_clarification": None,
                        "used_skills": _used_skills_meta,
                        "trace": trace,
                    }
            except Exception:
                pass  # fall through к обычному admit_ignorance

            reply = parsed.get(
                "reply",
                parsed.get(
                    "final_response", "Хм, я не знаю точного ответа. Может, поискать?"
                ),
            )
            # Сохраняем вопрос как pending для будущего поиска
            if owner_id is not None:
                try:
                    from src.core.memory.pending_questions import save_pending

                    await save_pending(owner_id, user_text[:500], "")
                except Exception:
                    pass
            return {
                "understood": "признаю незнание",
                "plan": [],
                "agents_to_call": [],
                "final_response": reply,
                "needs_clarification": None,
                "used_skills": _used_skills_meta,
                "trace": trace,
            }

        # ── plan_day handler ──
        if intent == "plan_day":
            # Автономный план дня: собираем всё и отдаём одним сообщением
            try:
                day_prompt = (
                    "Пользователь просит спланировать день. Собери информацию из доступных источников "
                    "и выдай ЕДИНЫМ сообщением (не диалогом, а сводкой):\n"
                    "1. Память: последние важные факты и контекст\n"
                    "2. Вопросы: неотвеченные pending вопросы\n"
                    "3. Напоминания: встречи, дедлайны\n"
                    "4. Новости: если есть\n"
                    "Формат: эмодзи-секции, кратко. Без 'давай проверим', без подтверждений."
                )
                resp = await provider.chat(
                    [
                        ChatMessage(role="system", content=day_prompt),
                        ChatMessage(role="user", content=user_text),
                    ],
                    task_type=TaskType.DEFAULT,
                )
                return {
                    "understood": "план дня",
                    "plan": [],
                    "agents_to_call": [],
                    "final_response": resp,
                    "needs_clarification": None,
                    "used_skills": _used_skills_meta,
                    "trace": trace,
                }
            except Exception:
                pass  # fallback к обычному admit_ignorance

        # ── Final response? ──
        if isinstance(parsed, dict) and "final_response" in parsed:
            final_response = parsed["final_response"]

            # Hallucination guard для final_response
            try:
                from src.core.intelligence.hallucination_guard import (
                    verify_claims,
                    apply_guard,
                )

                # Собираем memory_facts из контекста
                memory_facts = []
                if ctx and ctx.memory_context:
                    # Извлекаем факты из memory_context (это строка с фактами)
                    facts = [
                        f.strip("- ").strip()
                        for f in ctx.memory_context.split("\n")
                        if f.strip().startswith("-")
                    ]
                    memory_facts = [f for f in facts if len(f) > 10]

                if memory_facts:
                    verify_result = await verify_claims(
                        final_response, memory_facts, []
                    )
                    final_response, modified = apply_guard(
                        final_response, verify_result, confidence
                    )
            except Exception:
                pass  # best-effort

            return {
                "understood": parsed.get("understood", raw),
                "plan": parsed.get("plan", []),
                "agents_to_call": parsed.get("agents_to_call", []),
                "final_response": final_response,
                "needs_clarification": parsed.get("needs_clarification"),
                "used_skills": _used_skills_meta,
                "trace": trace,
            }

        # Fallback: return full parsed JSON (backward compat)
        parsed["used_skills"] = _used_skills_meta
        parsed["trace"] = trace
        return parsed

    # ── Max iterations exhausted ──
    logger.warning(
        "Maestro tool loop exhausted after %d iterations", MAX_TOOL_ITERATIONS
    )
    return {
        "understood": "tool loop exhausted",
        "plan": [],
        "agents_to_call": [],
        "final_response": "Я зациклился на вызове инструментов. Попробуй переформулировать запрос покороче.",
        "used_skills": _used_skills_meta,
        "trace": trace,
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
    "random": ("src.core.intelligence.maestro", "_invoke_random"),
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


async def _invoke_delegate(
    func: Any, provider, query: str, owner_id: int, **kwargs: Any
) -> dict:
    """Invokes a dynamic sub-agent with its own LLM call.

    The sub-agent receives a task description, optional instructions,
    and context, then runs an independent LLM analysis.

    Agent spec fields used:
        task (str): what to analyse (overrides query)
        instructions (str|None): custom system prompt additions
        context (str|None): additional data to analyse
        contact (str|None): optional contact to scope analysis
    """
    agent_spec = kwargs.get("agent_spec", {})
    task = agent_spec.get("task", query)
    instructions = agent_spec.get("instructions", "")
    context_data = agent_spec.get("context", "")

    system = (
        "Ты — аналитический суб-агент. Выполни анализ задачи и верни "
        "структурированный ответ. Будь точен, аргументирован и лаконичен."
    )
    if instructions:
        system += f"\n\nДополнительные инструкции:\n{instructions}"

    user_prompt = f"Задача: {task}"
    if context_data:
        user_prompt += f"\n\nКонтекст:\n{context_data}"

    try:
        from src.llm.base import ChatMessage

        raw = await asyncio.wait_for(
            provider.chat(
                [
                    ChatMessage(role="system", content=system),
                    ChatMessage(role="user", content=user_prompt),
                ],
                task_type=TaskType.DEFAULT,
            ),
            timeout=90.0,
        )
        return {
            "data": {"analysis": raw.strip(), "task": task},
            "success": True,
        }
    except Exception:
        logger.exception("delegate agent failed: %s", task)
        return {
            "data": {},
            "success": False,
            "error": "Sub-agent analysis failed",
        }


async def _invoke_random(
    func: Any, provider, query: str, owner_id: int, **kwargs: Any
) -> dict:
    """Random agent — handles non-standard/creative tasks with minimal prompting.

    Uses the default provider to respond to queries without domain-specific
    instructions. Suitable for creative, out-of-the-box, or unusual requests.
    """
    system = (
        "Ты — агент для нестандартных и творческих задач. "
        "Отвечай креативно, нестандартно, с юмором. "
        "Будь живым собеседником."
    )
    try:
        raw = await asyncio.wait_for(
            provider.chat(
                [
                    ChatMessage(role="system", content=system),
                    ChatMessage(role="user", content=query),
                ],
                task_type=TaskType.DEFAULT,
            ),
            timeout=60.0,
        )
        return {
            "data": {"response": raw.strip(), "query": query},
            "success": True,
        }
    except Exception:
        logger.exception("random agent failed: %s", query)
        return {
            "data": {},
            "success": False,
            "error": "Random agent failed",
        }


_AGENT_INVOKERS: dict[str, Any] = {
    "search": _invoke_search,
    "memory": _invoke_memory,
    "urgency": _invoke_urgency,
    "commitment": _invoke_commitment,
    "summarizer": _invoke_summarizer,
    "draft": _invoke_draft,
    "digest": _invoke_digest,
    "skill_creator": _invoke_skill_creator,
    "delegate": _invoke_delegate,
    "random": _invoke_random,
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

    # --- Special case: delegate (dynamic sub-agent) ---
    # The delegate agent doesn't need AGENT_REGISTRY entry or an import.
    # It creates its own LLM call via _invoke_delegate.
    if agent_type == "delegate":
        invoker = _AGENT_INVOKERS.get("delegate")
        if invoker is None:
            logger.error("Delegate invoker not found")
            return {
                "agent": "delegate",
                "data": {},
                "success": False,
                "error": "Delegate invoker not found",
            }
        try:
            result = await invoker(
                None, provider, query, owner_id, agent_spec=agent_spec
            )
            result["agent"] = "delegate"
            return result
        except Exception as e:
            logger.exception("Delegate agent failed")
            return {
                "agent": "delegate",
                "data": {},
                "success": False,
                "error": safe_str(e),
            }

    # --- Special case: random (non-standard/creative agent) ---
    if agent_type == "random":
        invoker = _AGENT_INVOKERS.get("random")
        if invoker is None:
            logger.error("Random invoker not found")
            return {
                "agent": "random",
                "data": {},
                "success": False,
                "error": "Random invoker not found",
            }
        try:
            result = await invoker(
                None, provider, query, owner_id, agent_spec=agent_spec
            )
            result["agent"] = "random"
            return result
        except Exception as e:
            logger.exception("Random agent failed")
            return {
                "agent": "random",
                "data": {},
                "success": False,
                "error": safe_str(e),
            }

    # --- Normal agent lookup ---
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
        return {"agent": agent_type, "data": {}, "success": False, "error": safe_str(e)}


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
    contact_id: int | None = None,
    userbot_manager: Any | None = None,
) -> dict[str, Any]:
    """Полный пайплайн: Maestro → агенты → финальный ответ.

    Args:
        contact_id: peer_id контакта, если пишем конкретному человеку.
                     Используется для инжекции per-contact правил.

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
        contact_id=contact_id,
        userbot_manager=userbot_manager,
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
        # Нет агентов и нет ответа — показываем понятные подсказки
        return {
            "final_response": sanitize_html(
                plan.get("final_response")
                or plan.get("needs_clarification")
                or FALLBACK_HINTS
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
        fallback_messages = [
            ChatMessage(role="system", content=MAESTRO_SYSTEM),
            ChatMessage(role="user", content=fallback_prompt),
        ]

        # Try streaming for fallback response
        stream = None
        try:
            stream = provider.chat_stream(fallback_messages, task_type=TaskType.MAESTRO)
        except (AttributeError, NotImplementedError):
            pass

        if stream is not None:
            return {
                "_stream": stream,
                "final_response": "",
                "plan": plan.get("plan", []),
                "used_agents": [],
                "agent_errors": agent_errors,
            }

        try:
            raw = await asyncio.wait_for(
                provider.chat(fallback_messages, task_type=TaskType.MAESTRO),
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
            if (
                "context_length" in safe_str(e).lower()
                or "token" in safe_str(e).lower()
            ):
                logger.warning("maestro fallback_request context overflow: %s", e)
                return {
                    "final_response": sanitize_html(
                        "📏 Контекст переполнен. Упрости запрос или уменьши историю."
                    ),
                    "plan": [],
                    "used_agents": [],
                    "agent_errors": agent_errors,
                }
            if "rate" in safe_str(e).lower():
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
        synthesis_messages = [
            ChatMessage(role="system", content=promo),
            ChatMessage(role="user", content=f"Пользователь сказал: {user_text}"),
        ]

        # Try streaming for final response
        stream = None
        try:
            stream = provider.chat_stream(
                synthesis_messages, task_type=TaskType.MAESTRO
            )
        except (AttributeError, NotImplementedError):
            pass

        if stream is not None:
            return {
                "_stream": stream,
                "final_response": "",
                "plan": plan.get("plan", []),
                "used_agents": used_agents,
                "agent_errors": agent_errors,
            }

        try:
            raw = await asyncio.wait_for(
                provider.chat(synthesis_messages, task_type=TaskType.MAESTRO),
                timeout=60.0,
            )
            raw = raw.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json|JSON)?\s*\n?", "", raw)
                raw = re.sub(r"\n?\s*```\s*$", "", raw)
            parsed = _extract_json_object(raw)
            if parsed is not None:
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
        "final_response": sanitize_html(plan.get("final_response") or FALLBACK_HINTS),
        "plan": plan.get("plan", []),
        "used_agents": used_agents,
        "agent_errors": agent_errors,
    }
