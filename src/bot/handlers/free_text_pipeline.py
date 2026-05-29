"""Pipeline stages for _process_text — extracted from free_text.py."""

import asyncio
import json
import logging
import re
import sys
import time
import uuid

from src.config import settings

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder

from src.bot.filters import OwnerOnly
from src.core.actions.action_guard import guard_intent
from src.core.actions.tool_registry import tool_registry
from src.core.actions.trajectory import actions_from_intent
from src.core.infra.key_guard import safe_str
from src.core.infra.task_manager import track_ff
from src.core.infra.text_sanitizer import sanitize_html
from src.core.intelligence.agent import route_intent
from src.core.intelligence.guardrails import evaluate as guardrail_evaluate
from src.core.intelligence.maestro import run_pipeline
from src.core.memory import conversation_context as ctx_store
from src.core.memory.memory_recall import recall
from src.db.repo import add_memory, get_or_create_user
from src.db.session import get_session
from src.userbot.manager import UserbotManager

from src.core.intelligence.pre_gate import check_pre_gate
from src.llm.base import ChatMessage, TaskType
from src.core.infra.timeutil import now_in_tz
from src.core.observability.response_trace import log_response_trace

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

from src.core.humanizer import (
    apply_anti_ai_mode,
    humanize_deep,
    analyze_ai_score,
    _cache_last_humanized,
    _preservation_check,
    normalize_anti_ai_mode,
)

from .free_text_exec import (
    exec_add_api_key,
    exec_add_news_topic,
    exec_add_reminder,
    exec_add_reminders_from_chat,
    exec_change_auto_mode,
    exec_check_memories,
    exec_classic_ask_chat,
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
    exec_link_memories,
    exec_show_memory_graph,
    exec_show_memory_health,
    exec_show_sessions,
    exec_show_suggestions,
    exec_store_memory,
    exec_toggle_api_key,
    exec_update_memory,
)

logger = logging.getLogger(__name__)

# ── Follow-up context ────────────────────────────────────────────────

_last_intent_ctx: dict[int, dict] = {}
_LAST_INTENT_TTL = 900.0

# ── Pending tool confirmations ────────────────────────────────────────
# Stores tool calls awaiting user confirmation (from maestro tool loop / guardrails).
# Format: {uid_str: {"telegram_id": int, "kind": "tool|intent", "tool": str,
#                    "tool_params": dict, "ts": float}}
_pending_confirmations: dict[str, dict] = {}
_pending_confirmations_lock = asyncio.Lock()
_PENDING_TTL = settings.pending_ttl_sec  # 5 минут — удаляем stale записи


async def _get_anti_ai_mode(owner_telegram_id: int) -> str:
    """Runtime mode for assistant responses: off/log/fix."""
    try:
        async with get_session() as session:
            owner = await get_or_create_user(session, owner_telegram_id)
            user_settings = getattr(owner, "settings", None)
            mode = getattr(user_settings, "anti_ai_mode", None)
            enabled = getattr(user_settings, "anti_ai_enabled", None)
            return normalize_anti_ai_mode(mode, enabled=enabled)
    except Exception:
        logger.debug("failed to load anti_ai_mode", exc_info=True)
        return "off"


async def _humanize_assistant_response(
    text: str,
    *,
    owner_telegram_id: int,
    context_hint: str | None,
    style_profile: str = "",
    source: str,
    mode: str | None = None,
) -> str:
    mode = mode or await _get_anti_ai_mode(owner_telegram_id)
    return apply_anti_ai_mode(
        text,
        mode=mode,
        context_hint=context_hint,
        style_profile=style_profile,
        user_id=owner_telegram_id,
        source=source,
    )


def _cleanup_stale_pending() -> None:
    """Remove entries older than ``_PENDING_TTL`` seconds."""
    now = time.monotonic()
    for uid in list(_pending_confirmations.keys()):
        entry = _pending_confirmations[uid]
        if now - entry.get("ts", 0) > _PENDING_TTL:
            del _pending_confirmations[uid]


def _confirm_tool_keyboard(uid: str) -> InlineKeyboardMarkup:
    """Inline-кнопки для подтверждения/отмены действия."""
    kb = InlineKeyboardBuilder()
    kb.row(
        InlineKeyboardButton(text="✅ Выполнить", callback_data=f"tool:confirm:{uid}"),
        InlineKeyboardButton(text="❌ Отмена", callback_data=f"tool:cancel:{uid}"),
    )
    return kb.as_markup()


def _redact_confirmation_params(params: dict) -> dict:
    redacted = {}
    sensitive = (
        "key",
        "token",
        "secret",
        "password",
        "credential",
        "value",
        "api_hash",
        "database_url",
        "proxy_url",
    )
    for name, value in params.items():
        if any(marker in str(name).lower() for marker in sensitive):
            redacted[name] = "***"
        else:
            redacted[name] = value
    return redacted


async def _store_tool_confirmation(
    telegram_id: int, tool: str, tool_params: dict
) -> str:
    """Сохраняет ожидающее подтверждение и возвращает uid для callback."""
    uid = uuid.uuid4().hex[:12]
    async with _pending_confirmations_lock:
        _cleanup_stale_pending()
        _pending_confirmations[uid] = {
            "telegram_id": telegram_id,
            "kind": "tool",
            "tool": tool,
            "tool_params": dict(tool_params),
            "ts": time.monotonic(),
        }
    return uid


async def _store_intent_confirmation(
    telegram_id: int, intent_name: str, intent: dict
) -> str:
    uid = uuid.uuid4().hex[:12]
    async with _pending_confirmations_lock:
        _cleanup_stale_pending()
        _pending_confirmations[uid] = {
            "telegram_id": telegram_id,
            "kind": "intent",
            "tool": intent_name,
            "tool_params": dict(intent),
            "ts": time.monotonic(),
        }
    return uid


async def _pop_tool_confirmation(uid: str, telegram_id: int) -> dict | None:
    """Извлекает и удаляет подтверждение. Возвращает None если не найдено."""
    async with _pending_confirmations_lock:
        _cleanup_stale_pending()
        pending = _pending_confirmations.pop(uid, None)
    if pending is None:
        return None
    if pending.get("telegram_id") != telegram_id:
        # Не совпадает владелец — кладём обратно
        async with _pending_confirmations_lock:
            _pending_confirmations[uid] = pending
        return None
    return pending


# ── Tool confirmation callback router ─────────────────────────────────

confirm_router = Router(name="free_text_tool_confirm")
confirm_router.callback_query.filter(OwnerOnly())


@confirm_router.callback_query(F.data.startswith("tool:confirm:"))
async def _cb_tool_confirm(
    callback: CallbackQuery,
    state: FSMContext,
    userbot_manager: UserbotManager,
) -> None:
    """Callback: пользователь подтвердил выполнение инструмента."""
    uid = callback.data.split(":", 2)[2]
    pending = await _pop_tool_confirmation(uid, callback.from_user.id)
    if pending is None:
        await callback.answer("⏳ Действие устарело или уже выполнено", show_alert=True)
        return

    tool_name = pending["tool"]
    tool_params = pending["tool_params"]
    logger.info(
        "User %d confirmed tool %s with params %s",
        callback.from_user.id,
        tool_name,
        _redact_confirmation_params(tool_params),
    )

    try:
        if pending.get("kind") == "intent":
            handler_info = INTENT_HANDLERS.get(
                tool_name
            ) or CLASSIC_INTENT_HANDLERS.get(tool_name)
            if handler_info is None:
                raise RuntimeError(f"Intent {tool_name!r} not found")
            handler, _ = handler_info
            await handler(
                tool_params,
                callback.message,
                state,
                userbot_manager,
                tz_name=tool_params.get("tz_name", "UTC"),
            )
            ok = True
        else:
            async with get_session() as session:
                owner = await get_or_create_user(session, callback.from_user.id)
                client = (
                    userbot_manager.get_client(callback.from_user.id)
                    if userbot_manager
                    else None
                )
                result = await tool_registry.execute(
                    tool_name,
                    _confirmed=True,
                    session=session,
                    user=owner,
                    client=client,
                    userbot_manager=userbot_manager,
                    **tool_params,
                )
            if isinstance(result, dict) and result.get("error"):
                raise RuntimeError(str(result["error"]))
            ok = result.get("ok", True) if isinstance(result, dict) else True
        if callback.message:
            if ok:
                await callback.message.edit_text(
                    sanitize_html(f"✅ {tool_name}: выполнено")
                )
            else:
                await callback.message.edit_text(
                    sanitize_html(f"⚠️ {tool_name}: выполнено с предупреждениями")
                )
        await callback.answer("✅ Выполнено")
    except Exception as e:
        logger.exception("Tool %s confirmation execution failed", tool_name)
        await callback.answer(f"❌ Ошибка: {safe_str(e)[:80]}", show_alert=True)
        if callback.message:
            await callback.message.edit_text(
                sanitize_html(f"❌ Ошибка при выполнении: {e}")
            )


@confirm_router.callback_query(F.data.startswith("tool:cancel:"))
async def _cb_tool_cancel(
    callback: CallbackQuery,
    state: FSMContext,
    userbot_manager: UserbotManager,
) -> None:
    """Callback: пользователь отменил выполнение инструмента."""
    uid = callback.data.split(":", 2)[2]
    async with _pending_confirmations_lock:
        _pending_confirmations.pop(uid, None)
    await callback.answer("❌ Отменено")
    if callback.message:
        await callback.message.edit_text("❌ Действие отменено.")


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
            raw_json = await provider.chat(
                [ChatMessage(role="user", content=prompt)], task_type=TaskType.DEFAULT
            )
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
    "update_memory": (h_adapter(exec_update_memory), "Обновить факт в памяти"),
    "link_memories": (h_adapter(exec_link_memories), "Связать два факта"),
    "show_memory_health": (h_adapter(exec_show_memory_health), "Здоровье памяти"),
    "show_memory_graph": (h_adapter(exec_show_memory_graph), "Граф памяти"),
    "show_sessions": (h_adapter(exec_show_sessions), "История сессий"),
    "show_suggestions": (h_adapter(exec_show_suggestions), "Паттерны памяти"),
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
    "ask_chat": (exec_classic_ask_chat, "Анализ чата"),
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

    # ── Risk-based guardrail check for HIGH/CRITICAL actions ────────
    if kind:
        gr = guardrail_evaluate(kind, intent, context={"is_new_contact": False})
        if gr.needs_confirm:
            intent_with_tz = dict(intent)
            intent_with_tz["tz_name"] = tz_name
            uid = await _store_intent_confirmation(
                message.from_user.id, kind, intent_with_tz
            )
            await safe_answer(
                message,
                sanitize_html(f"🤔 {gr.confirm_message}"),
                reply_markup=_confirm_tool_keyboard(uid),
            )
            _fire_record_trajectory(
                message.from_user.id,
                request_text=message.text or "",
                route_mode="dispatch_guard_confirm",
                intent_json=intent,
                actions_json=actions_from_intent(intent),
                success=True,
                error=None,
            )
            return

    handler_info = INTENT_HANDLERS.get(kind)
    if handler_info is not None:
        handler, _ = handler_info
        await handler(intent, message, state, userbot_manager, tz_name=tz_name)
        # Record action for smart correction
        try:
            from src.bot.handlers.smart_correction import record_action

            await record_action(
                message.from_user.id,
                {
                    "intent": kind,
                    "params": dict(intent),
                },
            )
        except Exception:
            logger.debug("record_action failed for %s", kind, exc_info=True)
        return
    await _execute_intent(intent, message, state, userbot_manager, tz_name=tz_name)
    # Record action for smart correction (classic intents)
    try:
        from src.bot.handlers.smart_correction import record_action

        await record_action(
            message.from_user.id,
            {
                "intent": kind,
                "params": dict(intent),
            },
        )
    except Exception:
        logger.debug("record_action failed for classic %s", kind, exc_info=True)


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


async def check_contact_rules(
    raw: str,
    owner_telegram_id: int,
    message: Message,
    userbot_manager: UserbotManager,
) -> bool:
    """Проверяет per-contact правила (например, «с Олей будь вежливее»).

    Возвращает True если обработано (early return).
    """
    try:
        from src.core.intelligence.adaptive_instructions import detect_contact_rule

        detected = await detect_contact_rule(raw)
        if not detected:
            return False

        contact_name = detected["contact_name"]
        rule_text = detected["rule"]

        # Разрешаем контакт по имени
        from src.core.contacts.contact_resolver import resolve
        from src.db.repo import get_or_create_user
        from src.db.session import get_session

        contact = None
        async with get_session() as session:
            owner = await get_or_create_user(session, owner_telegram_id)
            client = userbot_manager.get_client(owner_telegram_id)
            if client is None:
                logger.warning(
                    "check_contact_rules: no telethon client for user %s",
                    owner_telegram_id,
                )
                return False

            candidates = await resolve(
                client, owner, contact_name, limit=3, min_score=60
            )
            if candidates and candidates[0].score >= 60:
                contact = candidates[0]

        if not contact:
            await message.answer(
                sanitize_html(
                    f"🤔 Не могу найти контакт «{contact_name}» в твоей телефонной книге."
                )
            )
            return True

        # Сохраняем правило
        from src.core.contacts.contact_rules import add_contact_rule

        ok = await add_contact_rule(owner_telegram_id, contact.peer_id, rule_text)
        if ok:
            await message.answer(
                sanitize_html(
                    f"✅ Понял! Для контакта {contact.label()} буду соблюдать правило: «{rule_text}»."
                )
            )
        else:
            await message.answer(
                sanitize_html(f"⚠️ Не удалось сохранить правило для {contact.label()}.")
            )
        return True
    except Exception:
        logger.exception("check_contact_rules failed")
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
        # Record action for smart correction
        try:
            from src.bot.handlers.smart_correction import record_action

            await record_action(
                owner_telegram_id,
                {
                    "intent": intent.get("intent", ""),
                    "params": dict(intent),
                },
            )
        except Exception:
            logger.debug("record_action failed in followup", exc_info=True)
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


def _detect_context_hint(
    raw: str,
    plan_purpose: str | None = None,
) -> str | None:
    """Определяет контекстную подсказку для humanize_response.

    Сначала смотрит на purpose из плана авто-роутера,
    затем на ключевые слова в тексте пользователя.
    """
    # 1. Purpose → hint mapping
    purpose_map: dict[str, str] = {
        "search": "search",
        "analysis": "analysis",
        "draft": "send",
        "memory": "memory",
    }
    if plan_purpose and plan_purpose in purpose_map:
        return purpose_map[plan_purpose]

    # 2. Ключевые слова в тексте
    text_lower = raw.lower()
    if any(kw in text_lower for kw in ("найди", "поиск", "поищи", "search", "ищи")):
        return "search"
    if any(kw in text_lower for kw in ("проанализируй", "анализ", "разбери", "разбор")):
        return "analysis"
    if _looks_like_send_request(text_lower):
        return "send"
    if any(kw in text_lower for kw in ("напомни", "напоминание", "remind", "напомни")):
        return "reminder"
    if any(
        kw in text_lower
        for kw in ("запомни", "сохрани", "в память", "store_memory", "remember")
    ):
        return "memory"
    if any(
        kw in text_lower for kw in ("новости", "новость", "дайджест", "digest", "news")
    ):
        return "news"
    return None


def _looks_like_send_request(text_lower: str) -> bool:
    """True only for messaging intent, not generic "write text/code/recipe" requests."""
    if any(kw in text_lower for kw in ("отправь", "отправить", "сообщение", "с draft")):
        return True
    if "напиши" not in text_lower:
        return False
    recipient_markers = (
        " оле",
        " ему",
        " ей",
        " им ",
        " маме",
        " папе",
        " артёму",
        " артему",
    )
    return any(marker in f" {text_lower} " for marker in recipient_markers)


def _safe_for_deep_humanize(text: str, context_hint: str | None = None) -> bool:
    """Avoid second-pass LLM rewriting for structured or exact outputs."""
    if context_hint == "send":
        return False
    stripped = text.strip()
    if not stripped:
        return False
    structured_markers = ("```", "<code", "</", "{", "}", "[", "]")
    if any(marker in stripped for marker in structured_markers):
        return False
    if stripped.startswith(("{", "[", "- ", "* ", "1. ")):
        return False
    if "|" in stripped and "\n" in stripped:
        return False
    exact_output_words = (
        "json",
        "yaml",
        "sql",
        "код",
        "команд",
        "traceback",
        "exception",
    )
    return not any(word in stripped.lower() for word in exact_output_words)


async def execute_instant(
    plan,
    message: Message,
    raw: str,
    owner_telegram_id: int,
    turn_started: float,
    tz_name: str | None = None,
) -> bool:
    """Выполняет INSTANT-ответ (персонализированный). Возвращает True."""
    try:
        from src.core.infra.hooks import hooks

        await hooks.emit("on_message_received", user_id=owner_telegram_id, text=raw)
    except Exception:
        pass  # hooks are optional, never break core flow

    # Log user message to session (fire-and-forget)
    from src.core.scheduling.session_logger import log_user_message

    asyncio.ensure_future(log_user_message(message.from_user.id, raw))

    # ✨ Pre-LLM gate: handle greetings/farewells without LLM
    gate_response = check_pre_gate(raw)
    if gate_response:
        response = gate_response
        _cache_last_humanized(owner_telegram_id, response)
        await safe_answer(
            message, sanitize_html(response), reply_markup=memory_quick_keyboard()
        )
        await ctx_store.add_turn(message.from_user.id, raw[:200], response[:400])
        _fire_record_trajectory(
            owner_telegram_id,
            request_text=raw,
            route_mode="instant_gate",
            intent_json={"intent": "chat"},
            response_text=response,
            success=True,
            latency_ms=int((time.monotonic() - turn_started) * 1000),
        )
        await _post_turn_optimize(owner_telegram_id, raw, response)
        # Log assistant response to session
        from src.core.scheduling.session_logger import log_assistant_response

        asyncio.ensure_future(log_assistant_response(message.from_user.id, response))
        return True

    # Динамическое приветствие с учётом наличия памяти и сессии
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_telegram_id)
        memories = await recall(telegram_id=owner_telegram_id, limit=1, mode="normal")
        has_memory = bool(memories and memories.facts)
        has_session = owner.session is not None
        name = getattr(owner, "alias", None) or ""

    if not has_memory and not has_session:
        # Совершенно новый пользователь — first touch
        response = (
            "Привет! Я AI-ассистент. Обожаю своего создателя (@dutysissy)!\n\n"
            "Чтобы я заработал, нужно 3 шага:\n"
            "1. /login — привязать Telegram-аккаунт\n"
            "2. Добавить API-ключ для LLM (я подскажу как)\n"
            "3. /sync — я прочитаю твои чаты и запомню важное\n\n"
            "Поехали! Жми /login 🚀"
        )
    elif not has_memory:
        response = (
            "👋 <b>Привет! Я v3.0</b>\n\n"
            "Уже умею:\n"
            "🧠 Запоминать факты о тебе и контактах\n"
            "💬 Отвечать за тебя в ЛС (авто-ответ)\n"
            "📋 Вести список дел и напоминать\n"
            "📰 Собирать дайджест новостей\n"
            "🔍 Искать по истории переписок\n\n"
            "Чтобы я запомнил твои контакты и факты — жми /sync"
        )
    else:
        response = f"{_time_of_day_greeting(tz_name=tz_name)}{', ' + name if name else ''}! Чем займёмся?"

    # Humanize the assistant response according to Anti-AI runtime mode.
    context_hint = _detect_context_hint(
        raw, plan_purpose=plan.tasks[0].purpose.value if plan.tasks else None
    )
    response = await _humanize_assistant_response(
        response,
        owner_telegram_id=owner_telegram_id,
        context_hint=context_hint,
        source="free_text_pipeline.execute_instant",
    )
    _cache_last_humanized(owner_telegram_id, response)

    await safe_answer(message, sanitize_html(response))
    await ctx_store.add_turn(message.from_user.id, raw[:200], response[:400])
    # Record action for smart correction (skip first-time/intro messages)
    if has_memory or has_session:
        try:
            from src.bot.handlers.smart_correction import record_action

            await record_action(
                owner_telegram_id,
                {
                    "intent": "chat",
                    "params": {"reply": response[:200]},
                },
            )
        except Exception:
            logger.debug("record_action failed in execute_instant", exc_info=True)
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
    # Log assistant response to session
    from src.core.scheduling.session_logger import log_assistant_response

    asyncio.ensure_future(log_assistant_response(message.from_user.id, response))
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
            memory_context=getattr(plan, "memory_context", "") or None,
            user_id=owner_telegram_id,
        )
    except Exception as e:
        logger.exception("fast_route route_intent failed")
        plan.metrics["llm_ms"] = -1
        err_msg = safe_str(e)
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
        used_skills_json=intent.get("used_skills", []),
        response_text=_summarize_intent_for_memory(intent),
        success=True,
        latency_ms=int((time.monotonic() - turn_started) * 1000),
    )

    summary = _summarize_intent_for_memory(intent)
    await ctx_store.add_turn(message.from_user.id, raw, summary)
    try:
        if plan and plan.tasks:
            await ctx_store.set_last_purpose(
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
    try:
        from src.core.infra.hooks import hooks

        await hooks.emit("on_message_received", user_id=owner_telegram_id, text=raw)
    except Exception:
        pass  # hooks are optional, never break core flow

    # Log user message to session (fire-and-forget)
    from src.core.scheduling.session_logger import log_user_message

    asyncio.ensure_future(log_user_message(message.from_user.id, raw))

    # ✨ Pre-LLM gate: handle greetings/farewells without LLM
    gate_response = check_pre_gate(raw)
    if gate_response:
        response = gate_response
        _cache_last_humanized(owner_telegram_id, response)
        await safe_answer(
            message, sanitize_html(response), reply_markup=memory_quick_keyboard()
        )
        _fire_record_trajectory(
            owner_telegram_id,
            request_text=raw,
            route_mode="maestro_gate",
            intent_json={"intent": "chat"},
            response_text=response,
            success=True,
            latency_ms=int((time.monotonic() - turn_started) * 1000),
        )
        await ctx_store.add_turn(message.from_user.id, raw[:200], response[:400])
        await _post_turn_optimize(owner_telegram_id, raw, response)
        # Log assistant response to session
        from src.core.scheduling.session_logger import log_assistant_response

        asyncio.ensure_future(log_assistant_response(message.from_user.id, response))
        return True

    rag_needed = plan.recall_mode == "deep"
    # 📊 Productive thinking time: use the delay for better recall
    if not rag_needed and len(raw) > 30:
        try:
            from src.core.memory.contradiction_detector import detect_contradiction
            from src.core.memory.memory_recall import recall

            _deep = await recall(
                telegram_id=owner_telegram_id,
                query=raw,
                limit=5,
                mode="deep",
                include_deep=True,
            )
            _contra = await detect_contradiction(owner_telegram_id, raw)
            if _deep and _deep.facts and len(_deep.facts) > 3:
                rag_needed = True  # more facts found → upgrade to deep mode
                logger.debug(
                    "Upgraded to deep recall: %d facts found in thinking time",
                    len(_deep.facts),
                )
        except Exception:
            logger.debug("Enhanced recall in thinking time failed", exc_info=True)
    try:
        pipeline_result = await run_pipeline(
            provider,
            raw,
            owner_id=owner_telegram_id,
            history_block=history_block,
            memory_context=getattr(plan, "memory_context", "") or None,
            global_style=injected_style,
            self_profile=getattr(plan, "self_profile", "") or None,
            rag_enabled=rag_needed,
            contact_id=(
                plan.tasks[0].meta.get("contact_id")
                if getattr(plan, "tasks", None)
                and plan.tasks
                and getattr(plan.tasks[0], "meta", None)
                else None
            ),
            userbot_manager=userbot_manager,
        )

        # ── Handle tool confirmation needed ──────────────────────────
        if pipeline_result.get("confirmation_needed"):
            confirm_msg = pipeline_result.get(
                "confirm_message",
                pipeline_result.get("final_response", "Подтверди действие"),
            )
            tool_name = pipeline_result.get("tool", "")
            tool_params = pipeline_result.get("tool_params", {})
            uid = await _store_tool_confirmation(
                owner_telegram_id, tool_name, tool_params
            )
            await safe_answer(
                message,
                sanitize_html(f"🤔 {confirm_msg}"),
                reply_markup=_confirm_tool_keyboard(uid),
            )
            _fire_record_trajectory(
                owner_telegram_id,
                request_text=raw,
                route_mode="maestro_tool_confirm",
                intent_json={"intent": tool_name, **tool_params},
                actions_json=pipeline_result.get("plan", []),
                success=True,
                error=None,
                latency_ms=int((time.monotonic() - turn_started) * 1000),
            )
            await ctx_store.add_turn(
                message.from_user.id,
                raw[:200],
                f"[tool confirmation: {tool_name}]",
            )
            trace = dict(pipeline_result.get("trace") or {})
            log_response_trace(
                route="maestro_tool_confirm",
                owner_id=owner_telegram_id,
                memory_context=getattr(plan, "memory_context", "") or "",
                context_sources=trace.get("context_sources", []),
                tools_proposed=trace.get("tools_proposed", []),
                tools_executed=trace.get("tools_executed", []),
                tools_blocked=trace.get("tools_blocked", [tool_name]),
                guardrail_decision=trace.get("guardrail_decision", {}),
                humanizer_mode="off",
                humanizer_changed=False,
            )
            # Log assistant response to session
            from src.core.scheduling.session_logger import log_assistant_response

            asyncio.ensure_future(
                log_assistant_response(message.from_user.id, confirm_msg)
            )
            return True

        # ── Handle streaming response ────────────────────────────────
        stream = pipeline_result.get("_stream")
        if stream is not None:
            # Определяем context_hint заранее для финального humanize
            plan_purpose = plan.tasks[0].purpose.value if plan.tasks else None
            context_hint = _detect_context_hint(raw, plan_purpose=plan_purpose)

            # Получаем стилевой профиль
            try:
                from src.core.intelligence.style_matcher import (
                    get_or_update_style_profile,
                )

                style_block = await get_or_update_style_profile(owner_telegram_id)
            except Exception:
                style_block = None

            if settings.streaming_enabled:
                cursor = settings.streaming_cursor.strip()
                interval = settings.streaming_edit_interval

                # Send initial message with cursor
                sent_msg = await message.answer(cursor)
                full_text = ""
                last_update = asyncio.get_event_loop().time()

                try:
                    async for chunk in stream:
                        full_text += chunk
                        now = asyncio.get_event_loop().time()
                        if now - last_update >= interval:
                            display_text = full_text + settings.streaming_cursor
                            try:
                                await sent_msg.edit_text(display_text[:4000])
                            except Exception:
                                pass  # message deleted or too old
                            last_update = now
                except Exception:
                    logger.debug("Stream interrupted", exc_info=True)
            else:
                # Non-streaming: accumulate text silently
                sent_msg = await message.answer("⏳")
                full_text = ""
                try:
                    async for chunk in stream:
                        full_text += chunk
                except Exception:
                    logger.debug("Stream interrupted", exc_info=True)

            if not full_text.strip():
                try:
                    await sent_msg.edit_text("⚠️ Не получилось сгенерировать ответ")
                except Exception:
                    await message.answer("⚠️ Не получилось сгенерировать ответ")
                return True

            # Apply Anti-AI mode after streaming; off/log keep text unchanged.
            anti_ai_mode = await _get_anti_ai_mode(owner_telegram_id)
            original_text = full_text.strip()
            humanized = await _humanize_assistant_response(
                original_text,
                owner_telegram_id=owner_telegram_id,
                context_hint=context_hint,
                style_profile=style_block or "",
                source="free_text_pipeline.stream",
                mode=anti_ai_mode,
            )
            # Deep humanize if needed
            score, _ = analyze_ai_score(humanized)
            if (
                anti_ai_mode == "fix"
                and score > 0.3
                and len(humanized) > 100
                and _safe_for_deep_humanize(humanized, context_hint=context_hint)
            ):
                try:
                    humanized = await humanize_deep(
                        humanized, provider, user_style=style_block or ""
                    )
                except Exception:
                    logger.debug("humanize_deep failed on streamed text", exc_info=True)

            humanizer_changed = humanized != original_text
            response_text = sanitize_html(humanized)
            _cache_last_humanized(owner_telegram_id, response_text)

            # Final message without cursor
            try:
                await sent_msg.edit_text(
                    response_text[:4000], reply_markup=memory_quick_keyboard()
                )
            except Exception:
                await safe_answer(
                    message, response_text, reply_markup=memory_quick_keyboard()
                )

            # Auto-save facts
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

            _used_skills = pipeline_result.get("used_skills", [])
            trace = dict(pipeline_result.get("trace") or {})
            log_response_trace(
                route="maestro_stream",
                owner_id=owner_telegram_id,
                memory_context=getattr(plan, "memory_context", "") or "",
                context_sources=trace.get("context_sources", []),
                tools_proposed=trace.get("tools_proposed", []),
                tools_executed=trace.get("tools_executed", []),
                tools_blocked=trace.get("tools_blocked", []),
                guardrail_decision=trace.get("guardrail_decision", {}),
                humanizer_mode=anti_ai_mode,
                humanizer_changed=humanizer_changed,
                extra={"used_skills": _used_skills},
            )
            _fire_record_trajectory(
                owner_telegram_id,
                request_text=raw,
                route_mode="maestro",
                intent_json={"intent": "maestro"},
                actions_json=pipeline_result.get("plan", []),
                used_skills_json=_used_skills,
                response_text=response_text,
                success=True,
                error="; ".join(errors) if errors else None,
                latency_ms=int((time.monotonic() - turn_started) * 1000),
            )
            await ctx_store.add_turn(
                message.from_user.id, raw[:200], response_text[:400]
            )
            await _post_turn_optimize(owner_telegram_id, raw, response_text)
            try:
                from src.core.infra.hooks import hooks

                await hooks.emit(
                    "on_message_post_maestro",
                    user_id=owner_telegram_id,
                    input=raw,
                    response=response_text,
                    plan=pipeline_result.get("plan", []),
                )
            except Exception:
                pass
            # Log assistant response to session
            from src.core.scheduling.session_logger import log_assistant_response

            asyncio.ensure_future(
                log_assistant_response(message.from_user.id, response_text)
            )
            return True

        # ── Handle tool results from tool loop ───────────────────────
        # Если maestro вернул tool_result, используем его для обогащения
        # ответа, но итоговый ответ берём из final_response (LLM уже
        # синтезировала его с учётом результатов инструмента).
        response_text = pipeline_result.get("final_response", "")
        _used_skills = pipeline_result.get("used_skills", [])
        if response_text:
            # ── Humanizer: post-process response ──────────────────────
            # Определяем контекстную подсказку из purpose плана
            plan_purpose = plan.tasks[0].purpose.value if plan.tasks else None
            context_hint = _detect_context_hint(raw, plan_purpose=plan_purpose)

            # Получаем стилевой профиль пользователя
            try:
                from src.core.intelligence.style_matcher import (
                    get_or_update_style_profile,
                )

                style_block = await get_or_update_style_profile(owner_telegram_id)
            except Exception:
                style_block = None

            # Stage 1: Anti-AI mode (off/log/fix). Fix clips endings and applies light replacements.
            anti_ai_mode = await _get_anti_ai_mode(owner_telegram_id)
            original_response_text = response_text
            humanized = await _humanize_assistant_response(
                original_response_text,
                owner_telegram_id=owner_telegram_id,
                context_hint=context_hint,
                style_profile=style_block or "",
                source="free_text_pipeline.final_response",
                mode=anti_ai_mode,
            )

            # Stage 2: deep humanize if still too AI-like
            score, _ = analyze_ai_score(humanized)
            if (
                anti_ai_mode == "fix"
                and score > 0.3
                and len(humanized) > 100
                and _safe_for_deep_humanize(humanized, context_hint=context_hint)
            ):
                try:
                    user_style_hint = style_block or ""
                    humanized = await humanize_deep(
                        humanized, provider, user_style=user_style_hint
                    )
                except Exception:
                    logger.debug(
                        "humanize_deep failed, using light humanized", exc_info=True
                    )

            response_text = humanized
            _cache_last_humanized(owner_telegram_id, response_text)

            # ── Self-correction loop: re-generate if too AI-like ──────
            if anti_ai_mode == "fix" and response_text and len(response_text) > 50:
                for _ in range(2):
                    score_before, _ = analyze_ai_score(response_text)
                    if score_before < 0.3:
                        break
                    correction_prompt = (
                        f"Твой ответ вышел слишком AI-шаблонным (score={score_before:.2f}). "
                        f"Перепиши его естественно, как человек:\n\n{response_text[:1000]}"
                    )
                    try:
                        rewritten = await provider.chat(
                            [ChatMessage(role="user", content=correction_prompt)],
                            task_type=TaskType.HUMANIZE,
                        )
                        if rewritten and len(rewritten) > 20:
                            rewritten = _preservation_check(response_text, rewritten)
                            response_text = rewritten
                    except Exception:
                        break
            # ── End Self-correction ────────────────────────────────────

            # ── End Humanizer ─────────────────────────────────────────

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

            # If there's a tool_result that wasn't rendered into final_response
            # (edge case), append a brief note
            tool_result = pipeline_result.get("tool_result")
            extra_suffix = ""
            if tool_result and not response_text:
                extra_suffix = f"\n\n<code>⚙️ {json.dumps(tool_result, default=str, ensure_ascii=False)[:200]}</code>"

            trace = dict(pipeline_result.get("trace") or {})
            log_response_trace(
                route="maestro",
                owner_id=owner_telegram_id,
                memory_context=getattr(plan, "memory_context", "") or "",
                context_sources=trace.get("context_sources", []),
                tools_proposed=trace.get("tools_proposed", []),
                tools_executed=trace.get("tools_executed", []),
                tools_blocked=trace.get("tools_blocked", []),
                guardrail_decision=trace.get("guardrail_decision", {}),
                humanizer_mode=anti_ai_mode,
                humanizer_changed=response_text != original_response_text,
                extra={"used_skills": _used_skills},
            )
            await safe_answer(
                message,
                sanitize_html(response_text + extra_suffix),
                reply_markup=memory_quick_keyboard(),
            )
            _fire_record_trajectory(
                owner_telegram_id,
                request_text=raw,
                route_mode="maestro",
                intent_json={"intent": "maestro"},
                actions_json=pipeline_result.get("plan", []),
                used_skills_json=_used_skills,
                response_text=response_text,
                success=True,
                error="; ".join(errors) if errors else None,
                latency_ms=int((time.monotonic() - turn_started) * 1000),
            )
            await ctx_store.add_turn(
                message.from_user.id, raw[:200], response_text[:400]
            )
            await _post_turn_optimize(owner_telegram_id, raw, response_text)
            try:
                from src.core.infra.hooks import hooks

                await hooks.emit(
                    "on_message_post_maestro",
                    user_id=owner_telegram_id,
                    input=raw,
                    response=response_text,
                    plan=pipeline_result.get("plan", []),
                )
            except Exception:
                pass
            # Log assistant response to session
            from src.core.scheduling.session_logger import log_assistant_response

            asyncio.ensure_future(
                log_assistant_response(message.from_user.id, response_text)
            )
            return True
        return False
    except Exception:
        try:
            from src.core.infra.hooks import hooks

            await hooks.emit(
                "on_error",
                error=str(sys.exc_info()[1])
                if sys.exc_info()[1]
                else "maestro pipeline failed",
                context="free_text_pipeline.execute_maestro",
            )
        except Exception:
            pass  # hooks are optional, never break core flow
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
