"""Maestro — главный ИИ-координатор. Тяжёлая модель. Планирует и делегирует сабагентам."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from src.llm.base import ChatMessage


logger = logging.getLogger(__name__)

MAESTRO_SYSTEM = """Ты — главный ИИ-ассистент владельца Telegram-аккаунта. Ты управляешь командой специализированных агентов.

## Твои агенты
- **search** — находит контакты/чаты по имени (Оля → peer_id)
- **memory** — хранит и извлекает факты о людях, их предпочтения, прошлые темы
- **draft** — пишет черновики ответов
- **summarizer** — делает сводки переписок, catchup
- **digest** — собирает дайджест входящих сообщений
- **commitment** — извлекает обещания, дедлайны, договорённости
- **urgency** — классифицирует срочность сообщения (urgent/important/normal)

## Твоя задача
1. Понять что хочет пользователь
2. Определить каких агентов нужно вызвать
3. Собрать их ответы
4. Дать пользователю финальный ответ (на русском, лаконично)

## Формат ответа
Верни JSON:
{
  "understood": "что понял (1 фраза)",
  "plan": ["шаг1", "шаг2"],
  "agents_to_call": [
    {"agent": "search", "query": "что искать", "cache": true},
    {"agent": "memory", "query": "чей контекст", "cache": true}
  ],
  "final_response": "финальный ответ пользователю (если можешь ответить без агентов)",
  "needs_clarification": "вопрос к пользователю если что-то непонятно (или null)"
}

## Правила
- Если пользователь просто болтает («привет», «как дела») — НЕ вызывай агентов, ответь сам
- Если нужен контакт — вызови search
- Если нужен контекст о человеке — вызови memory
- Если нужно написать сообщение — вызови search + memory + draft
- Если вопрос про переписку — summarizer
- Будь лаконичен. Не переспрашивай если и так понятно.
"""

MAESTRO_AFTER_AGENTS = """Ты — главный ИИ-ассистент. Ты уже запросил информацию у своих агентов. Вот что они вернули:

{agent_results}

Теперь дай пользователю финальный ответ. Учти все данные от агентов.
Если агенты не нашли ничего полезного — скажи об этом.

Ответь JSON:
{{
  "final_response": "твой ответ пользователю (на русском, лаконично, учитывая все данные)"
}}"""


async def process(
    provider,  # LLMProvider
    user_text: str,
    *,
    history_block: str | None = None,
    memory_context: str | None = None,
    global_style: str | None = None,
) -> dict[str, Any]:
    """Главная точка входа. Maestro понимает пользователя и составляет план."""
    ctx_parts = []
    if memory_context:
        ctx_parts.append(f"Память о контактах:\n{memory_context}")
    if global_style:
        ctx_parts.append(f"Твой стиль общения:\n{global_style}")
    if history_block:
        ctx_parts.append(f"История диалога:\n{history_block}")

    context_str = "\n\n".join(ctx_parts) if ctx_parts else ""
    user_msg = (
        f"{context_str}\n\nПользователь: {user_text}"
        if context_str
        else f"Пользователь: {user_text}"
    )

    try:
        raw = await provider.chat(
            [
                ChatMessage(role="system", content=MAESTRO_SYSTEM),
                ChatMessage(role="user", content=user_msg),
            ],
            heavy=True,
        )
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
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
    session,  # AsyncSession
) -> dict:
    """Исполняет одного агента по спецификации из плана maestro."""
    agent_type = agent_spec.get("agent", "")
    query = agent_spec.get("query", "")
    cache = agent_spec.get("cache", True)
    cache_ttl = 300 if cache else 0

    try:
        if agent_type == "search":
            from src.agents.search_agent import resolve as search_resolve
            from src.db.repo import get_or_create_user, list_contacts

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
    provider, agents_to_call: list, *, owner_id: int, session
) -> list[dict]:
    """Запускает нескольких агентов параллельно."""
    if not agents_to_call:
        return []

    tasks = [
        _execute_agent(provider, spec, owner_id=owner_id, session=session)
        for spec in agents_to_call
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
) -> dict[str, Any]:
    """Полный пайплайн: Maestro → агенты → финальный ответ.

    Returns:
        dict с ключами:
          - final_response: str (всегда — текст для пользователя)
          - plan: list (план действий)
          - used_agents: list[str] (какие агенты сработали)
          - agent_errors: list[str] (ошибки агентов)
    """
    # --- Шаг 1: Maestro планирует ---
    plan = await process(
        provider,
        user_text,
        history_block=history_block,
        memory_context=memory_context,
        global_style=global_style,
    )

    used_agents = []
    agent_errors = []

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
        # Нет агентов и нет ответа — возвращаем сырой understood
        return {
            "final_response": plan.get("understood", "Не понял. Повтори."),
            "plan": plan.get("plan", []),
            "used_agents": [],
            "agent_errors": [],
        }

    from src.db.session import get_session

    async with get_session() as session:
        results = await _execute_agents_parallel(
            provider, agents_to_call, owner_id=owner_id, session=session
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
                lines = raw.split("\n")
                raw = "\n".join(
                    lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
                )
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
