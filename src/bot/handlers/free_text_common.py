"""Общие утилиты, кэш настроек, клавиатуры, post-turn optimization —
используются free_text.py, free_text_memory.py, free_text_settings.py."""

import asyncio
import logging
import time

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

from src.core.infra.timeutil import HM_RE, is_valid_tz, get_user_tz
from src.core.actions.trajectory import record_trajectory
from src.db.repo import get_or_create_user
from src.db.session import get_session

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings cache — однотенантный, TTL 30 сек, инвалидируется при /settings
# ---------------------------------------------------------------------------
_settings_cache: dict[str, object] | None = None  # type: ignore[assignment]
_settings_cache_ts: float = 0.0
_SETTINGS_CACHE_TTL: float = 30.0
_settings_lock = asyncio.Lock()


async def _get_owner_context(telegram_id: int) -> dict[str, object]:
    """Возвращает {owner_telegram_id, tz_name, use_heavy, global_style_profile} с TTL-кэшем."""
    global _settings_cache, _settings_cache_ts
    now = time.monotonic()
    if _settings_cache is not None and (now - _settings_cache_ts) < _SETTINGS_CACHE_TTL:
        return _settings_cache  # type: ignore[return-value]
    async with _settings_lock:
        # Double-check after acquiring lock (TOCTOU guard)
        now = time.monotonic()
        if (
            _settings_cache is not None
            and (now - _settings_cache_ts) < _SETTINGS_CACHE_TTL
        ):
            return _settings_cache  # type: ignore[return-value]
        async with get_session() as session:
            owner = await get_or_create_user(session, telegram_id)
            _settings_cache = {
                "owner_telegram_id": owner.telegram_id,
                "tz_name": get_user_tz(owner),
                "use_heavy": owner.settings.use_heavy_model if owner.settings else True,
                "global_style_profile": owner.global_style_profile,
            }
            _settings_cache_ts = now
        return _settings_cache  # type: ignore[return-value]


def invalidate_settings_cache() -> None:
    """Сбросить кэш настроек (вызывается при изменении /settings)."""
    global _settings_cache
    _settings_cache = None


def _fire_record_trajectory(*args: object, **kwargs: object) -> None:
    """Fire-and-forget запись траектории (не блокирует ответ пользователю)."""

    async def _safe() -> None:
        try:
            await record_trajectory(*args, **kwargs)  # type: ignore[arg-type]
        except Exception:
            logger.exception("fire-and-forget trajectory failed")

    asyncio.create_task(_safe())


def _coerce_setting_value(spec: str, raw):
    if spec == "bool":
        if isinstance(raw, bool):
            return raw, None
        if isinstance(raw, str) and raw.lower() in {"true", "yes", "on", "вкл", "1"}:
            return True, None
        if isinstance(raw, str) and raw.lower() in {"false", "no", "off", "выкл", "0"}:
            return False, None
        return None, "ожидаю true/false"
    if spec == "int":
        try:
            return int(raw), None
        except (TypeError, ValueError):
            return None, "ожидаю целое число"
    if spec == "str":
        if not isinstance(raw, str) or not raw.strip():
            return None, "ожидаю строку"
        return raw.strip(), None
    if spec == "hm":
        if isinstance(raw, str) and HM_RE.match(raw.strip()):
            return raw.strip(), None
        return None, "ожидаю время в формате HH:MM"
    if spec == "tz":
        if isinstance(raw, str) and is_valid_tz(raw.strip()):
            return raw.strip(), None
        return None, "не нашёл такой IANA timezone"
    if spec.startswith("choice:"):
        opts = set(spec[len("choice:") :].split(","))
        if isinstance(raw, str) and raw.strip() in opts:
            return raw.strip(), None
        return None, f"допустимые значения: {', '.join(sorted(opts))}"
    return None, "неизвестный тип"


def _confirm_keyboard(action_id: int):
    kb = InlineKeyboardBuilder()
    kb.row(
        InlineKeyboardButton(
            text="✅ Отправить", callback_data=f"send:confirm:{action_id}"
        ),
        InlineKeyboardButton(text="✏ Изменить", callback_data=f"send:edit:{action_id}"),
    )
    kb.row(
        InlineKeyboardButton(text="❌ Отмена", callback_data=f"send:cancel:{action_id}")
    )
    return kb.as_markup()


def _candidates_keyboard_send(candidates):
    kb = InlineKeyboardBuilder()
    for c in candidates:
        kb.row(
            InlineKeyboardButton(
                text=f"{c.label()} · {c.score}",
                callback_data=f"send:pick:{c.peer_id}",
            )
        )
    kb.row(InlineKeyboardButton(text="❌ Отмена", callback_data="send:cancel:0"))
    return kb.as_markup()


def _candidates_keyboard_chat(action: str, candidates):
    # action ∈ {summary, tasks, draft, catchup} — re-use chat:* callback'ов из chat_cmd
    kb = InlineKeyboardBuilder()
    for c in candidates:
        kb.row(
            InlineKeyboardButton(
                text=f"{c.label()} · {c.score}",
                callback_data=f"chat:{action}:{c.peer_id}",
            )
        )
    kb.row(InlineKeyboardButton(text="❌ Отмена", callback_data="chat:cancel:0"))
    return kb.as_markup()


def memory_quick_keyboard(contact_name: str = "") -> InlineKeyboardMarkup:
    """Inline-кнопки быстрых действий с памятью."""
    explain_cb = f"memq:explain:{contact_name}" if contact_name else "memq:explain:"
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🧠 Что помню", callback_data="memq:list"),
                InlineKeyboardButton(text="➕ Запомни", callback_data="memq:add"),
                InlineKeyboardButton(text="❌ Забудь", callback_data="memq:forget"),
                InlineKeyboardButton(text="🤔 Почему?", callback_data=explain_cb),
            ]
        ]
    )


def _summarize_intent_for_memory(intent: dict) -> str:
    # компактная запись «что я только что сделал» для памяти диалога
    kind = intent.get("intent")
    if kind == "multi":
        return "несколько действий: " + ", ".join(
            (a or {}).get("intent", "?") for a in intent.get("actions", [])[:5]
        )
    if kind == "send_message":
        return f"подготовил отправку «{(intent.get('text') or '')[:60]}» для {intent.get('recipient')}"
    if kind in {"summarize_chat", "tasks_for_chat", "draft_reply", "catchup"}:
        return f"{kind} с контактом {intent.get('contact')}"
    if kind == "find_in_chats":
        return f"искал в чатах: {intent.get('query')}"
    if kind == "news_digest":
        return f"новости: {intent.get('topic')}"
    if kind == "set_setting":
        return f"настройка {intent.get('key')} → {intent.get('value')}"
    if kind == "add_news_topic":
        return f"добавил тему: {intent.get('topic')}"
    if kind == "remove_news_topic":
        return f"убрал тему: {intent.get('topic')}"
    if kind == "add_reminder":
        return f"напоминание: {intent.get('text')}"
    if kind == "remove_reminder":
        return f"убрал напоминание: {intent.get('query')}"
    if kind == "add_reminders_from_chat":
        return f"вытащил обещания из чата с {intent.get('contact')}"
    if kind == "list_todos":
        return "показал список обещаний"
    if kind == "chat":
        return (intent.get("reply") or "")[:160]
    if kind == "store_memory":
        return "запомнил факт"
    if kind == "forget_memory":
        return "удалил из памяти"
    if kind == "list_memories":
        return "посмотрел память"
    if kind == "extract_memories_from_chat":
        return "извлёк факты из переписки"
    if kind == "check_memories":
        return "проверил актуальность памяти"
    if kind == "change_auto_mode":
        return "изменил авто-режим"
    if kind == "set_quiet_hours":
        return "настроил тихие часы"
    if kind == "show_inbox":
        return "посмотрел входящие"
    if kind == "full_analysis":
        return "запустил полный анализ"
    if kind == "clarify":
        return f"переспросил: {intent.get('question', '')[:100]}"
    return kind or ""


def _parse_iso_to_utc_naive(value, tz_name: str | None = None):
    if not value:
        return None
    try:
        from datetime import datetime, timezone
        from src.core.infra.timeutil import parse_tz

        s = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is not None:
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
        if tz_name:
            tz = parse_tz(tz_name)
            local_dt = dt.replace(tzinfo=tz)
            return local_dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        return None


# ---------------------------------------------------------------------------
# InstructionOptimizer integration — post-turn LLM review
# ---------------------------------------------------------------------------

# Rate-limit для post_turn_optimize: не чаще 1 раза в 5 минут на пользователя
_post_turn_last_call: dict[int, float] = {}
_post_turn_lock: "asyncio.Lock | None" = None
_post_turn_task: "asyncio.Task | None" = None


def _get_post_turn_lock() -> asyncio.Lock:
    global _post_turn_lock
    if _post_turn_lock is None:
        _post_turn_lock = asyncio.Lock()
    return _post_turn_lock


async def _post_turn_optimize(
    telegram_id: int,
    user_message: str,
    assistant_response: str,
) -> None:
    """
    Запускает LLM-ревью диалога через InstructionOptimizer.
    FIRE-AND-FORGET: не ждёт результат, rate-limited (1 раз в 5 мин).
    """
    if not user_message or not assistant_response:
        return

    now = time.monotonic()
    async with _get_post_turn_lock():
        # Cleanup: удаляем записи старше 1 часа
        stale = [uid for uid, ts in _post_turn_last_call.items() if now - ts > 3600]
        for uid in stale:
            del _post_turn_last_call[uid]
        if telegram_id in _post_turn_last_call:
            if now - _post_turn_last_call[telegram_id] < 300:
                return  # rate-limited
        _post_turn_last_call[telegram_id] = now

    async def _do_optimize():
        try:
            from src.db.session import get_session
            from src.db.repo import get_or_create_user
            from src.core.intelligence.instruction_optimizer import instruction_optimizer

            async with get_session() as session:
                owner = await get_or_create_user(session, telegram_id)
                await instruction_optimizer.post_turn_review(
                    session=session,
                    user_id=owner.id,
                    user_obj=owner,
                    user_message=user_message,
                    assistant_response=assistant_response,
                )
        except Exception:
            logger.debug("post_turn_optimize skipped", exc_info=True)

    global _post_turn_task
    if _post_turn_task and not _post_turn_task.done():
        _post_turn_task.cancel()
    _post_turn_task = asyncio.create_task(_do_optimize())
