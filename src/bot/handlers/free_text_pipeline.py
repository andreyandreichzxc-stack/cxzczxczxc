"""Pipeline stages for _process_text — extracted from free_text.py."""

import json
import logging
import re
import time

import asyncio
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from src.core.actions.action_guard import guard_intent
from src.core.actions.trajectory import actions_from_intent
from src.core.infra.task_manager import track_ff
from src.core.infra.text_sanitizer import sanitize_html
from src.core.intelligence.agent import route_intent
from src.core.intelligence.maestro import run_pipeline
from src.core.memory import conversation_context as ctx_store
from src.db.repo import add_memory, get_or_create_user
from src.db.session import get_session
from src.userbot.manager import UserbotManager

from src.llm.base import ChatMessage
from src.core.infra.timeutil import now_in_tz

from .free_text_common import (
    _fire_record_trajectory,
    _post_turn_optimize,
    _summarize_intent_for_memory,
    h_adapter,
    ht_adapter,
    hu_adapter,
    memory_quick_keyboard,
    safe_answer,
)
from src.core.intelligence.routing_wordlists import learn_routing as _learn_routing

from .free_text_exec import (
    exec_add_api_key,
    exec_add_news_topic,
    exec_add_reminder,
    exec_add_reminders_from_chat,
    exec_change_auto_mode,
    exec_check_memories,
    exec_classic_catchup,
    exec_classic_chat,
    exec_classic_draft_reply,
    exec_classic_find_in_chats,
    exec_classic_list_todos,
    exec_classic_news_digest,
    exec_classic_search,
    exec_classic_send_message,
    exec_classic_summarize_chat,
    exec_classic_tasks_for_chat,
    exec_classic_unknown,
    exec_clarify,
    exec_extract_memories,
    exec_forget_memory,
    exec_full_analysis,
    exec_index_chats,
    exec_list_keys,
    exec_list_memories,
    exec_remove_api_key,
    exec_remove_news_topic,
    exec_remove_reminder,
    exec_set_quiet_hours,
    exec_set_setting,
    exec_show_digest,
    exec_show_inbox,
    exec_show_profile,
    exec_show_self,
    exec_show_skills,
    exec_show_style,
    exec_show_threads,
    exec_show_today,
    exec_show_trajectory,
    exec_store_memory,
    exec_toggle_api_key,
)

logger = logging.getLogger(__name__)

# ── Follow-up context ────────────────────────────────────────────────

_last_intent_ctx: dict[int, dict] = {}
_LAST_INTENT_TTL = 900.0

_APPEND_KEYWORDS = ("добавь", "и ещё", "также", "кстати", "плюс", "ещё", "а ещё")
_REPLACE_KEYWORDS = ("нет", "лучше", "вместо", "точнее", "не так", "исправь", "поменяй")
_MULTI_KEYWORDS = ("и не забудь", "заодно", "и ещё")


# ── Auto-save facts about user ───────────────────────────────────────

_AUTO_SAVE_PROMPT = (
    "You are a fact extractor. Given a user message and assistant reply, "
    "extract ANY personal facts the user revealed about themselves. "
    "Only extract facts where the user explicitly states something about:\n"
    "- Personal details (name, birthday, job, location)\n"
    "- Preferences (likes, dislikes, habits)\n"
    "- Plans, commitments, goals\n"
    "- Relationships (family, friends, colleagues)\n"
    "- Experiences, events, memories\n\n"
    "Ignore general questions or requests. Return ONLY JSON:\n"
    '{"facts": [{"fact": "...", "sentiment": "positive|negative|neutral"}]} '
    'or {"facts": []} if no personal facts revealed. '
    "Fact must be a concise statement in third person (e.g. 'User works as a designer').\n\n"
    "User message: {user_text}\n"
    "Assistant reply: {assistant_text}"
)


async def _maybe_auto_save_facts(
    user_text: str,
    response_text: str,
    telegram_id: int,
    provider,
) -> None:
    """Fire-and-forget: LLM извлекает личные факты → сохраняет в память."""
    # Quick pre-check: skip if message is clearly not personal
    text_lower = user_text.lower()
    if not any(
        kw in text_lower
        for kw in (
            "я ",
            "мой ",
            "моя ",
            "моё ",
            "мои ",
            "мне ",
            "меня ",
            "у меня",
            "день рождения",
            "др ",
            "работаю",
            "учусь",
            "живу",
            "люблю",
            "нравится",
            "хочу",
            "планирую",
            "собираюсь",
            "занимаюсь",
        )
    ):
        return

    async def _do_save():
        try:
            prompt = _AUTO_SAVE_PROMPT.format(
                user_text=user_text[:500].replace("{", "{{").replace("}", "}}"),
                assistant_text=response_text[:300]
                .replace("{", "{{")
                .replace("}", "}}"),
            )
            # NOTE: LLMProvider.chat() doesn't accept max_tokens;
            # the prompt asks for short JSON, so the LLM will return a compact response
            # with the provider's default max_tokens.
            raw_json = await provider.chat([ChatMessage(role="user", content=prompt)])
            # Parse LLM response as JSON
            import json as _json

            cleaned = raw_json.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```[a-z]*\s*|\s*```$", "", cleaned).strip()
            facts_data = _json.loads(cleaned)
            facts_list = facts_data.get("facts", [])
            if not facts_list:
                return

            async with get_session() as session:
                owner = await get_or_create_user(session, telegram_id)
                stored = 0
                for f in facts_list:
                    fact_text = f.get("fact", "").strip()
                    if not fact_text or len(fact_text) < 5:
                        continue
                    sentiment = f.get("sentiment", "neutral")
                    if sentiment not in ("positive", "negative", "neutral"):
                        sentiment = "neutral"
                    await add_memory(
                        session,
                        owner,
                        fact=fact_text,
                        contact_id=None,
                        sentiment=sentiment,
                        source="auto",
                    )
                    stored += 1
                if stored:
                    logger.info(
                        "Auto-saved %d facts for user %d: %s",
                        stored,
                        telegram_id,
                        "; ".join(f["fact"][:50] for f in facts_list),
                    )
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.debug("Auto-save facts skipped", exc_info=True)

    track_ff(asyncio.create_task(_do_save()))


def _save_intent_context(tg_id: int, intent: dict) -> None:
    _last_intent_ctx[tg_id] = {
        "intent": intent,
        "expires_at": time.monotonic() + _LAST_INTENT_TTL,
    }


def _detect_followup(raw: str, tg_id: int) -> tuple[dict, str] | None:
    """Если raw — продолжение предыдущего intent'а, вернуть (модифицированный intent, update_type).
    update_type: "append", "replace", "multi_add". Возвращает None если не продолжение."""
    entry = _last_intent_ctx.get(tg_id)
    if not entry or time.monotonic() > entry["expires_at"]:
        _last_intent_ctx.pop(tg_id, None)
        return None
    prev = entry["intent"]
    stripped = raw.strip().lower()
    words = stripped.split()[:3]
    first3 = " ".join(words)

    for kw in _REPLACE_KEYWORDS:
        if first3.startswith(kw):
            new_text = raw.strip()
            for kw2 in _REPLACE_KEYWORDS:
                if new_text.lower().startswith(kw2):
                    new_text = new_text[len(kw2) :].strip(", ").strip()
                    break
            modified = dict(prev)
            if "text" in modified:
                modified["text"] = new_text
            elif "query" in modified:
                modified["query"] = new_text
            return (modified, "replace")

    for kw in _APPEND_KEYWORDS:
        if first3.startswith(kw):
            new_text = raw.strip()
            for kw2 in _APPEND_KEYWORDS:
                if new_text.lower().startswith(kw2):
                    new_text = new_text[len(kw2) :].strip(", ").strip()
                    break
            modified = dict(prev)
            if "text" in modified:
                modified["text"] = modified.get("text", "") + " " + new_text
            elif "query" in modified:
                modified["query"] = modified.get("query", "") + " " + new_text
            return (modified, "append")

    for kw in _MULTI_KEYWORDS:
        if first3.startswith(kw):
            new_text = raw.strip()
            for kw2 in _MULTI_KEYWORDS:
                if new_text.lower().startswith(kw2):
                    new_text = new_text[len(kw2) :].strip(", ").strip()
                    break
            new_intent = {
                "intent": prev.get("intent", "chat"),
                "text": new_text,
            }
            return (new_intent, "multi_add")

    return None


async def _execute_intent(
    intent, message, state, userbot_manager, *, tz_name: str
) -> None:
    kind = intent.get("intent")
    handler_info = CLASSIC_INTENT_HANDLERS.get(kind)
    if handler_info is not None:
        handler, _ = handler_info
        await handler(intent, message, state, userbot_manager, tz_name=tz_name)
        return
    await message.answer("❓ Неизвестный intent.")


# ── Intent handler registries ────────────────────────────────────────

INTENT_HANDLERS: dict[str, tuple[callable, str]] = {
    "set_setting": (h_adapter(exec_set_setting), "Изменить настройку"),
    "add_news_topic": (h_adapter(exec_add_news_topic), "Добавить новостную тему"),
    "remove_news_topic": (h_adapter(exec_remove_news_topic), "Удалить новостную тему"),
    "add_reminder": (ht_adapter(exec_add_reminder), "Добавить напоминание"),
    "remove_reminder": (h_adapter(exec_remove_reminder), "Удалить напоминание"),
    "add_reminders_from_chat": (
        hu_adapter(exec_add_reminders_from_chat),
        "Извлечь напоминания из чата",
    ),
    "store_memory": (h_adapter(exec_store_memory), "Сохранить в память"),
    "forget_memory": (h_adapter(exec_forget_memory), "Удалить из памяти"),
    "list_memories": (h_adapter(exec_list_memories), "Показать память"),
    "extract_memories_from_chat": (
        hu_adapter(exec_extract_memories),
        "Извлечь воспоминания из чата",
    ),
    "check_memories": (h_adapter(exec_check_memories), "Проверить память"),
    "change_auto_mode": (h_adapter(exec_change_auto_mode), "Сменить авто-режим"),
    "set_quiet_hours": (h_adapter(exec_set_quiet_hours), "Установить тихие часы"),
    "show_inbox": (hu_adapter(exec_show_inbox), "Показать входящие"),
    "show_self": (h_adapter(exec_show_self), "Показать свой профиль"),
    "full_analysis": (h_adapter(exec_full_analysis), "Полный анализ"),
    "add_api_key": (h_adapter(exec_add_api_key), "Добавить API-ключ"),
    "remove_api_key": (h_adapter(exec_remove_api_key), "Удалить API-ключ"),
    "toggle_api_key": (h_adapter(exec_toggle_api_key), "Включить/выключить ключ"),
    "list_keys": (h_adapter(exec_list_keys), "Показать ключи"),
    "show_digest": (h_adapter(exec_show_digest), "Показать дайджест"),
    "show_today": (h_adapter(exec_show_today), "Показать сегодня"),
    "show_skills": (h_adapter(exec_show_skills), "Показать навыки"),
    "show_threads": (h_adapter(exec_show_threads), "Показать треды"),
    "show_trajectory": (h_adapter(exec_show_trajectory), "Показать траекторию"),
    "show_style": (h_adapter(exec_show_style), "Показать стиль"),
    "show_profile": (h_adapter(exec_show_profile), "Показать профиль"),
    "index_chats": (h_adapter(exec_index_chats), "Переиндексировать чаты"),
    "clarify": (h_adapter(exec_clarify), "Уточнить"),
}

CLASSIC_INTENT_HANDLERS: dict[str, tuple[callable, str]] = {
    "chat": (exec_classic_chat, "Чат"),
    "unknown": (exec_classic_unknown, "Неизвестный"),
    "list_todos": (exec_classic_list_todos, "Список задач"),
    "send_message": (exec_classic_send_message, "Отправить сообщение"),
    "search": (exec_classic_search, "Поиск"),
    "find_in_chats": (exec_classic_find_in_chats, "Поиск по чатам"),
    "news_digest": (exec_classic_news_digest, "Новостной дайджест"),
    "summarize_chat": (exec_classic_summarize_chat, "Саммари чата"),
    "tasks_for_chat": (exec_classic_tasks_for_chat, "Задачи из чата"),
    "draft_reply": (exec_classic_draft_reply, "Черновик ответа"),
    "catchup": (exec_classic_catchup, "Где остановились"),
}


# ── Dispatch ─────────────────────────────────────────────────────────


async def _dag_dispatch(
    sub_intents: list[dict],
    message: Message,
    state: FSMContext,
    userbot_manager: UserbotManager,
    *,
    tz_name: str,
) -> None:
    """DAG-диспетчер: независимые sub-intents выполняются параллельно.

    Формат sub_intent:
      {"intent": "...", ..., "depends_on": [0, 2]}
      depends_on — список индексов в sub_intents, которые должны выполниться ДО этого.
      Если depends_on отсутствует или пуст — действие считается независимым.

    При циклических зависимостях — fallback на последовательное выполнение.
    """
    if not sub_intents:
        await message.answer("Не понял, что сделать.")
        return

    n = len(sub_intents)
    if n == 1:
        await _dispatch(
            sub_intents[0], message, state, userbot_manager, tz_name=tz_name
        )
        return

    # Build dependency graph (Kahn's algorithm)
    in_degree = [0] * n
    children: list[list[int]] = [[] for _ in range(n)]
    has_any_dep = False

    for i, sub in enumerate(sub_intents):
        deps = sub.get("depends_on") or []
        if deps:
            has_any_dep = True
        for d in deps:
            if isinstance(d, int) and 0 <= d < n and d != i:
                children[d].append(i)
                in_degree[i] += 1

    # Если ни у одного sub-intent нет depends_on — все независимы → параллельно
    if not has_any_dep:
        tasks = [
            _dispatch(sub, message, state, userbot_manager, tz_name=tz_name)
            for sub in sub_intents
        ]
        await _run_dag_level(tasks, sub_intents)
        return

    # Topo-sort by levels
    level: list[int] = [i for i in range(n) if in_degree[i] == 0]
    levels: list[list[int]] = []
    visited = 0

    while level:
        levels.append(level)
        visited += len(level)
        next_level: list[int] = []
        for node in level:
            for child in children[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    next_level.append(child)
        level = next_level

    if visited < n:
        # Cycle detected — fallback to sequential
        logger.warning(
            "DAG cycle detected in multi-intent (%d/%d visited), "
            "falling back to sequential",
            visited,
            n,
        )
        for sub in sub_intents:
            try:
                await _dispatch(sub, message, state, userbot_manager, tz_name=tz_name)
            except Exception:
                logger.exception(
                    "DAG fallback: sub-intent %s failed", sub.get("intent", "?")
                )
        return

    # Execute per level in parallel
    for level_indices in levels:
        tasks = [
            _dispatch(sub_intents[i], message, state, userbot_manager, tz_name=tz_name)
            for i in level_indices
        ]
        await _run_dag_level(tasks, sub_intents, level_indices)


async def _run_dag_level(
    tasks: list,
    sub_intents: list[dict],
    indices: list[int] | None = None,
) -> None:
    """Запускает группу sub-intents параллельно, логирует ошибки."""
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            i = indices[idx] if indices else idx
            logger.error(
                "Sub-intent %d (%s) failed: %s",
                i,
                sub_intents[i].get("intent", "?"),
                result,
            )


async def _dispatch(intent, message, state, userbot_manager, *, tz_name: str) -> None:
    guard = guard_intent(intent)
    if not guard.allowed:
        _fire_record_trajectory(
            message.from_user.id,
            request_text=message.text or "",
            route_mode="dispatch_guard",
            intent_json=intent if isinstance(intent, dict) else None,
            actions_json=actions_from_intent(
                intent if isinstance(intent, dict) else None
            ),
            success=False,
            error=guard.reason,
        )
        await message.answer(
            sanitize_html(f"⚠️ Действие остановлено guardrail: {guard.reason}")
        )
        return
    intent = guard.intent
    kind = intent.get("intent")
    handler_info = INTENT_HANDLERS.get(kind)
    if handler_info is not None:
        handler, _ = handler_info
        await handler(intent, message, state, userbot_manager, tz_name=tz_name)
        return
    await _execute_intent(intent, message, state, userbot_manager, tz_name=tz_name)


# ── Pipeline stages ──────────────────────────────────────────────────


async def check_instructions(
    raw: str, owner_telegram_id: int, message: Message
) -> bool:
    """Проверяет adaptive instructions. Возвращает True если обработано (early return)."""
    try:
        from src.core.intelligence.adaptive_instructions import (
            detect_instruction,
            apply_instruction,
        )

        instr = await detect_instruction(raw, owner_telegram_id)
        if instr:
            from src.db.models import InstructionCandidate, InstructionEvent

            async with get_session() as session:
                owner_db = await get_or_create_user(session, owner_telegram_id)
                event = InstructionEvent(
                    user_id=owner_db.id,
                    raw_text=raw[:500],
                    detected_rule=instr["rule"],
                    action=instr["action"],
                )
                session.add(event)
                if instr["is_safe"]:
                    await message.answer(
                        sanitize_html(f"✅ Понял! Больше не буду {instr['rule']}.")
                    )
                    await apply_instruction(owner_telegram_id, instr["rule"])
                    await session.flush()
                    return True
                else:
                    candidate = InstructionCandidate(
                        user_id=owner_db.id,
                        rule=instr["rule"],
                        category=instr["category"],
                        is_safe=False,
                        llm_reviewed=False,
                    )
                    session.add(candidate)
                    await session.flush()
                    await message.answer(
                        sanitize_html(
                            f"🤔 Понял: «{instr['rule']}». Применить это правило? (да/нет)"
                        )
                    )
                    return True
    except Exception:
        logger.exception("adaptive instruction check failed")
    return False


async def check_persona(raw: str, owner_telegram_id: int, message: Message) -> bool:
    """Проверяет adaptive persona.

    Два режима:
    1. Явная команда (detect_persona_change) — блокирует дальнейшую обработку,
       бот подтверждает изменение.
    2. Авто-адаптация (auto_adapt_from_context) — НЕ блокирует, работает тихо
       в фоне: анализирует настроение и плавно корректирует стиль.
    """
    try:
        from src.core.intelligence.adaptive_persona import (
            detect_persona_change,
            apply_persona_changes,
            auto_adapt_from_context,
        )

        # Явная команда: пользователь сказал «короче» / «дружелюбнее»
        change = await detect_persona_change(raw)
        if change:
            await apply_persona_changes(owner_telegram_id, change["changes"])
            await message.answer(sanitize_html(f"✅ Понял! Буду {change['reason']}."))
            return True

        # Авто-адаптация: бот сам чувствует настроение
        # Не блокирует — возвращает False, чтобы сообщение обрабатывалось дальше
        try:
            await auto_adapt_from_context(owner_telegram_id, raw, provider=None)
        except Exception:
            logger.debug("auto_adapt_from_context failed", exc_info=True)

    except Exception:
        logger.exception("adaptive persona check failed")
    return False


async def check_followup(
    raw: str,
    owner_telegram_id: int,
    message: Message,
    state: FSMContext,
    userbot_manager: UserbotManager,
    tz_name: str,
    turn_started: float,
) -> bool:
    """Проверяет follow-up контекст. Возвращает True если обработано."""
    followup = _detect_followup(raw, owner_telegram_id)
    if followup:
        intent, _update_type = followup
        await _execute_intent(intent, message, state, userbot_manager, tz_name=tz_name)
        _save_intent_context(owner_telegram_id, intent)
        _fire_record_trajectory(
            owner_telegram_id,
            request_text=raw,
            route_mode="followup",
            intent_json=intent,
            success=True,
            latency_ms=int((time.monotonic() - turn_started) * 1000),
        )
        return True
    return False


def _time_of_day_greeting(tz_name: str | None = None) -> str:
    """Возвращает приветствие в зависимости от времени суток пользователя."""
    hour = now_in_tz(tz_name).hour
    if 6 <= hour < 12:
        return "Доброе утро"
    if 12 <= hour < 18:
        return "Добрый день"
    if 18 <= hour < 23:
        return "Добрый вечер"
    return "Доброй ночи"


async def execute_instant(
    plan,
    message: Message,
    raw: str,
    owner_telegram_id: int,
    turn_started: float,
    tz_name: str | None = None,
) -> bool:
    """Выполняет INSTANT-ответ (персонализированный). Возвращает True."""
    response = plan.final_response
    # Персонализация: добавляем имя + время суток в приветствия
    user_name = message.from_user.first_name
    if user_name and response:
        greeting_markers = (
            "Привет",
            "Здаров",
            "Хай",
            "Ку",
            "Hello",
            "Hi",
        )
        for marker in greeting_markers:
            if response.startswith(marker):
                tod = _time_of_day_greeting(tz_name=tz_name)
                if tod not in response:
                    response = f"{tod}, {user_name}! {response}"
                else:
                    response = f"{response}, {user_name}"
                break
    await safe_answer(message, sanitize_html(response))
    ctx_store.add_turn(message.from_user.id, raw[:200], response[:400])
    _fire_record_trajectory(
        owner_telegram_id,
        request_text=raw,
        route_mode="instant",
        intent_json={"intent": "chat"},
        response_text=plan.final_response,
        success=True,
        latency_ms=int((time.monotonic() - turn_started) * 1000),
    )
    await _post_turn_optimize(owner_telegram_id, raw, plan.final_response)
    return True


async def execute_fast_route(
    raw: str,
    plan,
    provider,
    message: Message,
    state: FSMContext,
    userbot_manager: UserbotManager,
    tz_name: str,
    owner_telegram_id: int,
    history_block: str,
    turn_started: float,
    now_local_str: str,
) -> bool:
    """Выполняет FAST_ROUTE. Возвращает True."""
    fast_start = time.monotonic()
    try:
        intent = await route_intent(
            provider,
            raw,
            heavy=False,
            now_local=now_local_str,
            tz_name=tz_name,
            history_block=history_block,
            memory_context=plan.memory_context,
            user_id=owner_telegram_id,
        )
    except Exception as e:
        logger.exception("fast_route route_intent failed")
        plan.metrics["llm_ms"] = -1
        err_msg = str(e)
        _fire_record_trajectory(
            owner_telegram_id,
            request_text=raw,
            route_mode="fast_route",
            intent_json={"intent": "chat"},
            success=False,
            error=err_msg[:4000],
            latency_ms=int((time.monotonic() - turn_started) * 1000),
        )
        if len(err_msg) > 300:
            err_msg = err_msg[:300] + "…"
        await safe_answer(
            message,
            sanitize_html(
                f"❌ Ошибка при обработке запроса.\n\n"
                f"<code>{err_msg}</code>\n\n"
                "<i>Если ошибка повторяется — проверь ключ в /settings → 🔑 API-ключи "
                "и модель в /settings → 🤖 LLM.</i>"
            ),
        )
        return True

    plan.metrics["llm_ms"] = int((time.monotonic() - fast_start) * 1000)
    plan.metrics["total_ms"] = plan.metrics.get("recall_ms", 0) + plan.metrics.get(
        "llm_ms", 0
    )
    logger.info("Fast route metrics: %s", json.dumps(plan.metrics, default=str))

    # Проверка confidence — если низкий, уточняем
    if await _check_intent_confidence(intent, message):
        return True

    if intent.get("intent") == "multi":
        actions = intent.get("actions") or []
        if not isinstance(actions, list) or not actions:
            await message.answer("Не понял, что сделать.")
            return True
        await _dag_dispatch(actions, message, state, userbot_manager, tz_name=tz_name)
    elif "intents" in intent:
        await _dag_dispatch(
            intent["intents"], message, state, userbot_manager, tz_name=tz_name
        )
    else:
        await _dispatch(intent, message, state, userbot_manager, tz_name=tz_name)

    # Learning Router: запоминаем ключевые слова из успешных интентов
    # (только для action-интентов, фильтр внутри learn_routing)
    intent_kind = intent.get("intent", "")
    if intent_kind not in ("multi",):
        _learn_routing(raw, intent_kind)
    elif intent_kind == "multi":
        for sub in intent.get("actions", intent.get("intents", [])):
            _learn_routing(raw, sub.get("intent", ""))

    _save_intent_context(owner_telegram_id, intent)

    _fire_record_trajectory(
        owner_telegram_id,
        request_text=raw,
        route_mode="fast_route",
        intent_json=intent,
        actions_json=actions_from_intent(intent),
        response_text=_summarize_intent_for_memory(intent),
        success=True,
        latency_ms=int((time.monotonic() - turn_started) * 1000),
    )

    summary = _summarize_intent_for_memory(intent)
    ctx_store.add_turn(message.from_user.id, raw, summary)
    try:
        if plan and plan.tasks:
            ctx_store.set_last_purpose(
                message.from_user.id, plan.tasks[0].purpose.value
            )
    except Exception:
        logger.exception("failed to set last purpose")
    return True


async def execute_maestro(
    raw: str,
    plan,
    provider,
    message: Message,
    state: FSMContext,
    userbot_manager: UserbotManager,
    tz_name: str,
    owner_telegram_id: int,
    history_block: str,
    turn_started: float,
    injected_style: str | None = None,
) -> bool:
    """Выполняет MAESTRO pipeline. Возвращает True если обработано, False для fallback."""
    rag_needed = plan.recall_mode == "deep"
    try:
        pipeline_result = await run_pipeline(
            provider,
            raw,
            owner_id=owner_telegram_id,
            history_block=history_block,
            memory_context=plan.memory_context,
            global_style=injected_style,
            rag_enabled=rag_needed,
        )
        response_text = pipeline_result.get("final_response", "")
        if response_text:
            # Auto-save: fire-and-forget сохранение фактов о пользователе
            track_ff(
                asyncio.create_task(
                    _maybe_auto_save_facts(
                        raw, response_text, owner_telegram_id, provider
                    )
                )
            )
            used = pipeline_result.get("used_agents", [])
            errors = pipeline_result.get("agent_errors", [])
            if used:
                logger.debug("Maestro agents: %s", used)
            if errors:
                logger.debug("Maestro agent errors: %s", errors)
            await safe_answer(
                message,
                sanitize_html(response_text),
                reply_markup=memory_quick_keyboard(),
            )
            _fire_record_trajectory(
                owner_telegram_id,
                request_text=raw,
                route_mode="maestro",
                intent_json={"intent": "maestro"},
                actions_json=pipeline_result.get("plan", []),
                response_text=response_text,
                success=True,
                error="; ".join(errors) if errors else None,
                latency_ms=int((time.monotonic() - turn_started) * 1000),
            )
            ctx_store.add_turn(message.from_user.id, raw[:200], response_text[:400])
            await _post_turn_optimize(owner_telegram_id, raw, response_text)
            return True
        return False
    except Exception:
        logger.debug("Maestro pipeline failed, falling back to route_intent")
        return False


async def _check_intent_confidence(intent: dict, message: Message) -> bool:
    """Проверяет confidence. Если низкий — уточняет и возвращает True (обработано)."""
    # Если нет поля confidence — считаем что уверен (backward compat)
    if "confidence" not in intent:
        return False
    confidence = intent.get("confidence", 1.0)
    if not isinstance(confidence, (int, float)):
        return False
    if confidence >= 0.6:
        return False

    question = intent.get("question") or "Не совсем понял. Что именно сделать?"
    await message.answer(sanitize_html(f"🤔 {question}"))
    return True
