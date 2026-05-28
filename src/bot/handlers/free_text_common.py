"""Общие утилиты, кэш настроек, клавиатуры, post-turn optimization —
используются free_text.py, free_text_memory.py, free_text_settings.py."""

import asyncio
import logging
import re
import time

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

from src.bot.reply_dedup import dedup
from src.config import settings
from src.core.infra.formatting import auto_format

# ── Telegram message length safety ─────────────────────────────────────
TELEGRAM_SAFE_MAX = settings.safe_message_length  # Telegram hard limit is 4096 chars


def _smart_split(text: str, max_len: int = TELEGRAM_SAFE_MAX) -> list[str]:
    """Split text into chunks respecting paragraph then sentence boundaries.

    Never splits mid-word. Sent as multiple messages if too long.
    """
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    # First pass: split on paragraph boundaries
    paragraphs = text.split("\n\n")
    buf = ""

    for para in paragraphs:
        if not buf:
            buf = para
            continue
        candidate = buf + "\n\n" + para
        if len(candidate) <= max_len:
            buf = candidate
        else:
            chunks.append(buf)
            buf = para

    if buf:
        chunks.append(buf)

    # Second pass: hard-split any chunks still over the limit
    result: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_len:
            result.append(chunk)
        else:
            result.extend(_hard_split(chunk, max_len))
    return result


def _hard_split(text: str, max_len: int) -> list[str]:
    """Hard-split a single chunk at sentence boundaries, never mid-word."""
    parts: list[str] = []
    while text:
        if len(text) <= max_len:
            parts.append(text)
            break
        # Try to find a sentence boundary within the limit
        chunk = text[:max_len]
        last_period = chunk.rfind(". ")
        last_newline = chunk.rfind("\n")
        split_at = max(last_period, last_newline)
        if split_at > max_len // 2:
            parts.append(text[: split_at + 1].strip())
            text = text[split_at + 1 :].strip()
        else:
            # No good boundary found — hard cut at the last space
            last_space = chunk.rfind(" ")
            if last_space > max_len // 2:
                parts.append(text[:last_space].strip())
                text = text[last_space:].strip()
            else:
                parts.append(chunk.strip())
                text = text[max_len:].strip()
    return parts


async def safe_answer(
    message, text: str, max_len: int = TELEGRAM_SAFE_MAX, **kwargs
) -> None:
    """Send ``text`` via ``message.answer()``, splitting into multiple messages if too long.
    ``reply_markup`` (if any) is attached only to the last message.
    """
    # Reaction engine: short responses → emoji reaction instead of text
    _REACTION_MAP = {
        "👍": ["ок", "ладно", "хорошо", "принято", "да", "ага", "угу"],
        "👎": ["нет", "не", "не надо", "отмена", "не так"],
        "❤️": ["спасибо", "благодарю", "отлично", "супер", "круто"],
        "😢": ["жаль", "грустно", "печаль", "сочувствую"],
        "😡": ["бесит", "злюсь", "раздражён"],
        "🎉": ["ура", "поздравляю", "йоу"],
    }
    if len(text) < 50 and "```" not in text:
        text_lower = text.lower().rstrip(".!,?;: \n")
        for emoji, triggers in _REACTION_MAP.items():
            if text_lower in triggers:
                try:
                    from aiogram.types import ReactionTypeEmoji

                    await message.react([ReactionTypeEmoji(emoji=emoji)])
                    return  # Don't send text
                except Exception:
                    break  # Fall through to normal text send

    if dedup.is_duplicate(message.chat.id, text):
        return
    # Apply auto-formatting for Telegram HTML
    text = auto_format(text)
    parts = _smart_split(text, max_len)
    for i, part in enumerate(parts):
        final_kwargs = {}
        if i == len(parts) - 1:
            final_kwargs = kwargs  # reply_markup only on the last chunk
        await message.answer(part, **final_kwargs)


from src.core.infra.task_manager import track_ff
from src.core.infra.timeutil import HM_RE, is_valid_tz, get_user_tz
from src.core.actions.trajectory import record_trajectory
from src.db.repo import get_or_create_user
from src.db.session import get_session

logger = logging.getLogger(__name__)

# ── Model name validation ─────────────────────────────────────────────
_MODEL_NAME_RE = re.compile(r"^[a-zA-Z0-9@/_.:-]{1,128}$")


# ---------------------------------------------------------------------------
# Settings cache — per-user, TTL 30 сек, инвалидируется при /settings
# ---------------------------------------------------------------------------
_settings_cache: dict[int, dict] = {}  # telegram_id → cached context
_settings_cache_ts: dict[int, float] = {}  # telegram_id → last cache time
_SETTINGS_CACHE_TTL: float = 30.0
_settings_lock = asyncio.Lock()


async def _get_owner_context(telegram_id: int) -> dict[str, object]:
    """Возвращает {owner_telegram_id, tz_name, use_heavy, global_style_profile} с TTL-кэшем (per-user)."""
    now = time.monotonic()
    cached_ts = _settings_cache_ts.get(telegram_id, 0.0)
    if telegram_id in _settings_cache and (now - cached_ts) < _SETTINGS_CACHE_TTL:
        return _settings_cache[telegram_id]  # type: ignore[return-value]
    async with _settings_lock:
        # Double-check after acquiring lock (TOCTOU guard)
        now = time.monotonic()
        cached_ts = _settings_cache_ts.get(telegram_id, 0.0)
        if telegram_id in _settings_cache and (now - cached_ts) < _SETTINGS_CACHE_TTL:
            return _settings_cache[telegram_id]  # type: ignore[return-value]
        async with get_session() as session:
            owner = await get_or_create_user(session, telegram_id)
            ctx = {
                "owner_telegram_id": owner.telegram_id,
                "tz_name": get_user_tz(owner),
                "use_heavy": owner.settings.use_heavy_model if owner.settings else True,
                "global_style_profile": owner.global_style_profile,
            }
            _settings_cache[telegram_id] = ctx
            _settings_cache_ts[telegram_id] = now
        return ctx  # type: ignore[return-value]


async def invalidate_settings_cache(telegram_id: int | None = None) -> None:
    """Сбросить кэш настроек (вызывается при изменении /settings).
    Если telegram_id=None — сбрасывает весь кэш."""
    async with _settings_lock:
        if telegram_id is not None:
            _settings_cache.pop(telegram_id, None)
            _settings_cache_ts.pop(telegram_id, None)
        else:
            _settings_cache.clear()
            _settings_cache_ts.clear()


def _fire_record_trajectory(*args: object, **kwargs: object) -> None:
    """Fire-and-forget запись траектории (не блокирует ответ пользователю)."""

    async def _safe() -> None:
        try:
            await record_trajectory(*args, **kwargs)  # type: ignore[arg-type]
        except Exception:
            logger.exception("fire-and-forget trajectory failed")

    track_ff(asyncio.create_task(_safe()))


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
    if spec == "model":
        if not isinstance(raw, str):
            return None, "ожидаю строку (имя модели)"
        val = raw.strip()
        if not val or val.lower() in ("default", "по умолчанию", "сброс", "сбросить"):
            return "", None  # clear override
        if len(val) > 128:
            return None, "имя модели слишком длинное (макс. 128)"
        if not _MODEL_NAME_RE.match(val):
            return None, (
                "недопустимые символы в имени модели. "
                "Допустимы: буквы, цифры, @ / _ . : -"
            )
        return val, None
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


KIND_EMOJI = {"user": "👤", "group": "👥", "channel": "📢", "bot": "🤖"}


def _group_candidates(candidates: list, max_display: int = 8):
    """Group candidates by peer_kind, return (displayed_list, hidden_count)."""
    if len(candidates) <= max_display:
        return candidates, 0

    # Sort by score descending, show top max_display
    sorted_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
    return sorted_candidates[:max_display], len(sorted_candidates) - max_display


def _candidates_keyboard_send(candidates):
    kb = InlineKeyboardBuilder()
    displayed, hidden = _group_candidates(candidates, max_display=8)

    # Group by kind with visual separator
    shown_kinds: set[str] = set()
    for c in displayed:
        emoji = KIND_EMOJI.get(c.peer_kind, "•")
        if c.peer_kind not in shown_kinds:
            shown_kinds.add(c.peer_kind)
        kb.row(
            InlineKeyboardButton(
                text=f"{emoji} {c.label()} · {c.score}",
                callback_data=f"send:pick:{c.peer_id}",
            )
        )

    if hidden:
        kb.row(
            InlineKeyboardButton(
                text=f"🔍 Ещё {hidden} контактов — уточните имя",
                callback_data="send:cancel:0",  # cancel just dismisses
            )
        )
    kb.row(InlineKeyboardButton(text="❌ Отмена", callback_data="send:cancel:0"))
    return kb.as_markup()


def _candidates_keyboard_chat(action: str, candidates):
    # action ∈ {summary, tasks, draft, catchup} — re-use chat:* callback'ов из chat_cmd
    kb = InlineKeyboardBuilder()
    displayed, hidden = _group_candidates(candidates, max_display=8)

    shown_kinds: set[str] = set()
    for c in displayed:
        emoji = KIND_EMOJI.get(c.peer_kind, "•")
        if c.peer_kind not in shown_kinds:
            shown_kinds.add(c.peer_kind)
        kb.row(
            InlineKeyboardButton(
                text=f"{emoji} {c.label()} · {c.score}",
                callback_data=f"chat:{action}:{c.peer_id}",
            )
        )

    if hidden:
        kb.row(
            InlineKeyboardButton(
                text=f"🔍 Ещё {hidden} контактов — уточните имя",
                callback_data="chat:cancel:0",
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
    if kind == "update_memory":
        return "обновил факт в памяти"
    if kind == "link_memories":
        return "связал два факта в памяти"
    if kind == "show_memory_health":
        return "посмотрел здоровье памяти"
    if kind == "show_memory_graph":
        return "посмотрел граф памяти"
    if kind == "show_sessions":
        return "посмотрел историю сессий"
    if kind == "show_suggestions":
        return "посмотрел паттерны памяти"
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
# Dispatch adapters — приводят хендлеры к единой сигнатуре
# ---------------------------------------------------------------------------


def h_adapter(fn):
    """Адаптер: fn(intent, message) → унифицированная dispatch-сигнатура."""

    async def _w(intent, message, state, userbot_manager, *, tz_name):
        return await fn(intent, message)

    return _w


def hu_adapter(fn):
    """Адаптер: fn(intent, message, userbot_manager) → унифицированная."""

    async def _w(intent, message, state, userbot_manager, *, tz_name):
        return await fn(intent, message, userbot_manager)

    return _w


def ht_adapter(fn):
    """Адаптер: fn(intent, message, *, tz_name) → унифицированная."""

    async def _w(intent, message, state, userbot_manager, *, tz_name):
        return await fn(intent, message, tz_name=tz_name)

    return _w


# ---------------------------------------------------------------------------
# InstructionOptimizer integration — post-turn LLM review
# ---------------------------------------------------------------------------

# Rate-limit для post_turn_optimize: не чаще 1 раза в 5 минут на пользователя
_post_turn_last_call: dict[int, float] = {}
_post_turn_lock: "asyncio.Lock | None" = None
_post_turn_tasks: "dict[int, asyncio.Task]" = {}


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
            from src.core.intelligence.instruction_optimizer import (
                instruction_optimizer,
            )

            async with get_session() as session:
                owner = await get_or_create_user(session, telegram_id)
                await instruction_optimizer.post_turn_review(
                    session=session,
                    user_id=owner.id,
                    user_obj=owner,
                    user_message=user_message,
                    assistant_response=assistant_response,
                )
        except (Exception, asyncio.CancelledError):
            logger.debug("post_turn_optimize skipped", exc_info=True)

    existing = _post_turn_tasks.get(telegram_id)
    if existing and not existing.done():
        existing.cancel()
    _post_turn_tasks[telegram_id] = asyncio.create_task(_do_optimize())
