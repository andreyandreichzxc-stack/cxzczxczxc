"""Свободный текст (и голос) → агент → действие. Регистрируется последним в bot/app.py,
чтобы команды и FSM перехватывали свои события раньше."""

import asyncio
import json
import logging
import time
from pathlib import Path

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    InlineKeyboardButton,
    Message,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder

from src.bot.filters import OwnerOnly
from src.config import settings
from src.core.intelligence.agent import route_intent
from src.core.contacts.chat_service import load_chat
from src.core.intelligence.maestro import run_pipeline
from src.core.intelligence.smart_autorouter import make_plan, RoutePurpose, ResponseMode
from src.core.actions.action_guard import guard_intent
from src.core.actions.commitment_extractor import extract_and_save_commitments
from src.core.contacts.contact_resolver import resolve, resolve_with_llm
from src.core.scheduling.news import build_news_digest
from src.core.intelligence.summarizer import catchup, draft_reply, summarize_chat
from src.core.memory import conversation_context as ctx_store
from src.core.infra.text_sanitizer import sanitize_html
from src.core.infra.timeutil import (
    fmt_local,
    now_in_tz,
)
from src.core.infra.transcription import transcription_service
from src.db.models import LlmKeySlot
from src.db.repo import (
    add_commitment,
    add_key_slot,
    get_api_key,
    get_contact,
    get_contact_profile,
    get_or_create_user,
    get_self_profile,
    list_key_slots,
    list_open_commitments,
    update_commitment_status,
    upsert_contact,
)
from src.db.session import get_session
from src.llm.router import _ensure_utc, build_provider
from src.core.actions.trajectory import actions_from_intent, record_trajectory
from src.core.intelligence.skills import build_skill_index, record_skill_usages
from src.userbot import get_active_telethon_client, get_userbot_manager
from src.userbot.manager import UserbotManager

from .free_text_common import (
    _candidates_keyboard_chat,
    _candidates_keyboard_send,
    _confirm_keyboard,
    _fire_record_trajectory,
    _get_owner_context,
    _parse_iso_to_utc_naive,
    _post_turn_optimize,
    _summarize_intent_for_memory,
    memory_quick_keyboard,
)
from .free_text_memory import (
    _exec_check_memories,
    _exec_extract_memories,
    _exec_forget_memory,
    _exec_list_memories,
    _exec_store_memory,
)
from .free_text_settings import (
    _exec_add_news_topic,
    _exec_change_auto_mode,
    _exec_remove_news_topic,
    _exec_set_quiet_hours,
    _exec_set_setting,
)


logger = logging.getLogger(__name__)
router = Router(name="free_text")
router.message.filter(OwnerOnly())


# Voice transcription queue (non-blocking background processing)
_voice_queue: asyncio.Queue = asyncio.Queue(maxsize=20)
_voice_worker_task: asyncio.Task | None = None


def start_voice_worker() -> asyncio.Task:
    """Запустить фонового worker'а для транскрипции голоса (если ещё не запущен).

    Вызывается при старте приложения (main.py).
    """
    global _voice_worker_task
    if _voice_worker_task is None or _voice_worker_task.done():
        _voice_worker_task = asyncio.create_task(
            _voice_worker(), name="voice-transcription-worker"
        )
    return _voice_worker_task


async def stop_voice_worker() -> None:
    """Остановить voice worker (graceful shutdown)."""
    global _voice_worker_task
    if _voice_worker_task and not _voice_worker_task.done():
        _voice_worker_task.cancel()
        try:
            await _voice_worker_task
        except asyncio.CancelledError:
            pass
        _voice_worker_task = None
        logger.info("Voice transcription worker stopped")


async def _voice_worker() -> None:
    """Фоновый обработчик очереди голосовой транскрипции.

    Бесконечный цикл: забирает задание из очереди, транскрибирует,
    чистит файл, отвечает пользователю и передаёт текст в _process_text.
    При крахе одной задачи не падает — логирует и идёт дальше.
    """
    while True:
        try:
            job = await _voice_queue.get()
            (
                voice_path,
                message,
                state,
                userbot_manager,
                file_unique_id,
                mode,
                api_provider,
                openai_key,
                gemini_key,
                mistral_key,
            ) = job

            try:
                text = await transcription_service.transcribe(
                    voice_path,
                    file_id=file_unique_id,
                    mode=mode,
                    openai_key=openai_key,
                    gemini_key=gemini_key,
                    mistral_key=mistral_key,
                    api_provider=api_provider,
                )
            except Exception:
                logger.exception("voice transcription failed in worker")
                try:
                    await message.answer("❌ Не удалось распознать голосовое.")
                except Exception:
                    logger.exception("failed to send error message from worker")
                finally:
                    _cleanup_voice_file(voice_path)
                continue

            _cleanup_voice_file(voice_path)

            text = (text or "").strip()
            if not text:
                try:
                    await message.answer("🎙 Не услышал текста в этом сообщении.")
                except Exception:
                    logger.exception("failed to send empty transcription message")
                continue

            try:
                await message.answer(f"🎙 <i>Услышал:</i> {text}")
            except Exception:
                logger.exception("failed to send transcription result")

            try:
                await _process_text(text, message, state, userbot_manager)
            except Exception:
                logger.exception("Failed to process transcribed text in worker")

        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("Voice worker error")
        finally:
            _voice_queue.task_done()


def _cleanup_voice_file(voice_path: Path) -> None:
    """Безопасно удалить временный файл голосового сообщения."""
    try:
        voice_path.unlink(missing_ok=True)
    except Exception:
        logger.debug("cleanup voice file failed: %s", voice_path, exc_info=True)


CHAT_LOAD_LIMIT = 50

# Follow-up context: remembers last intent for 60 seconds to detect continuations
_last_intent_ctx: dict[
    int, dict
] = {}  # telegram_id → {"intent": dict, "expires_at": float}
_LAST_INTENT_TTL = 60.0  # seconds

_APPEND_KEYWORDS = ("добавь", "и ещё", "также", "кстати", "плюс", "ещё", "а ещё")
_REPLACE_KEYWORDS = ("нет", "лучше", "вместо", "точнее", "не так", "исправь", "поменяй")
_MULTI_KEYWORDS = ("и не забудь", "заодно", "и ещё")


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
    # Проверяем первые 3 слова
    words = stripped.split()[:3]
    first3 = " ".join(words)

    # REPLACE: "нет", "лучше", ...
    for kw in _REPLACE_KEYWORDS:
        if first3.startswith(kw):
            new_text = raw.strip()
            # Убираем ключевое слово из начала
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

    # APPEND: "добавь", "и ещё", ...
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

    # MULTI: "и не забудь", "заодно"
    for kw in _MULTI_KEYWORDS:
        if first3.startswith(kw):
            new_text = raw.strip()
            for kw2 in _MULTI_KEYWORDS:
                if new_text.lower().startswith(kw2):
                    new_text = new_text[len(kw2) :].strip(", ").strip()
                    break
            # Возвращаем intent с извлечённым текстом как новый intent
            new_intent = {
                "intent": prev.get("intent", "chat"),
                "text": new_text,
            }
            return (new_intent, "multi_add")

    # Если не нашли ключевых слов — не follow-up
    return None


async def _execute_intent(
    intent, message, state, userbot_manager, *, tz_name: str
) -> None:
    kind = intent.get("intent")
    client = userbot_manager.get_client(message.from_user.id)

    # selectin-loaded settings/api_keys доступны после закрытия сессии
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        provider = await build_provider(session, owner)
        heavy = owner.settings.use_heavy_model

    if kind == "chat":
        reply = sanitize_html(intent.get("reply"))
        if not reply:
            reply = "Готов помочь. Уточни, пожалуйста."
        await message.answer(reply)
        return

    if kind == "unknown" or kind is None:
        await message.answer(
            "🤷 Не понял, что нужно сделать. Я умею: писать сообщения людям, делать саммари переписок, "
            "извлекать задачи, ловить «где мы остановились», искать по сообщениям, собирать новостной "
            "дайджест по теме, показывать обещания. Попробуй сформулировать иначе или открой /help."
        )
        return

    if kind == "list_todos":
        async with get_session() as session:
            owner = await get_or_create_user(session, message.from_user.id)
            items = await list_open_commitments(session, owner)
        if not items:
            await message.answer("🎉 Открытых обязательств нет")
            return
        from src.core.infra.timeutil import fmt_local

        lines = []
        for c in items[:30]:
            who = "Я" if c.direction == "mine" else (c.peer_name or "Они")
            d = fmt_local(c.deadline_at, tz_name)
            lines.append(f"• <b>{who}</b>: {c.text} (до {d})")
        await message.answer(
            f"📋 Открытых обязательств: <b>{len(items)}</b>\n\n" + "\n".join(lines)
        )
        return

    if client is None:
        await message.answer("Сначала /login — нужен подключённый Telegram-аккаунт.")
        return

    if kind == "send_message":
        recipient = (intent.get("recipient") or "").strip()
        text = (intent.get("text") or "").strip()
        if not recipient or not text:
            await message.answer("🤷 Не хватает кому/что отправить. Уточни.")
            return
        candidates = await resolve_with_llm(client, owner, recipient, provider)
        if not candidates:
            await message.answer(
                sanitize_html(f"Не нашёл контакт «{recipient}». Попробуй /sync.")
            )
            return
        if len(candidates) == 1 or candidates[0].score >= 90:
            target = candidates[0]
            ctx_store.set_last_peer(
                message.from_user.id, target.peer_id, target.display_name
            )
            payload = json.dumps(
                {"peer_id": target.peer_id, "text": text}, ensure_ascii=False
            )
            async with get_session() as session:
                owner = await get_or_create_user(session, message.from_user.id)
                from src.db.repo import create_pending_action

                action = await create_pending_action(
                    session, user_id=owner.id, kind="send_message", payload=payload
                )
                guard_hint = ""
                if target.peer_id:
                    try:
                        from src.core.contacts.send_guard import build_send_guard

                        guard = await build_send_guard(
                            owner.telegram_id, target.peer_id, text
                        )
                        if guard.formatted_html:
                            guard_hint = "\n\n" + guard.formatted_html
                    except Exception:
                        logger.warning("send guard failed", exc_info=True)

            await message.answer(
                sanitize_html(
                    f"🤔 <b>Готов отправить</b>\n\n"
                    f"→ <b>Кому:</b> {target.label()}\n"
                    f"→ <b>Текст:</b>\n{text}{guard_hint}"
                ),
                reply_markup=_confirm_keyboard(action.id),
            )
        else:
            await state.set_data({"send_text": text})
            await message.answer(
                sanitize_html(f"Кому именно отправить «<i>{text[:80]}</i>»?"),
                reply_markup=_candidates_keyboard_send(candidates),
            )
        return

    if kind == "search":
        query = (intent.get("query") or "").strip()
        peer_query = (intent.get("peer_query") or intent.get("contact") or "").strip()
        if not query:
            await message.answer("Не понял, что искать.")
            return
        await message.answer(sanitize_html(f"🔎 Ищу: <i>{query}</i>…"))
        # Если нет явного контакта — cmd_search сам сделает cross_chat_search (FTS)
        from src.bot.handlers.search import cmd_search
        from aiogram.filters import CommandObject

        await cmd_search(
            message,
            CommandObject(prefix="/", command="search", args=query),
            userbot_manager,
        )
        return

    if kind == "find_in_chats":
        query = (intent.get("query") or "").strip()
        action = (intent.get("action") or "catchup").strip()
        if action not in {"catchup", "summary", "tasks", "draft"}:
            action = "catchup"
        if not query:
            await message.answer("🤷 Не понял, по какой теме искать.")
            return
        await message.answer(sanitize_html(f"🔎 Ищу по моим чатам: «<i>{query}</i>»…"))
        await _find_chats_and_offer(message, client, query, action)
        return

    if kind == "news_digest":
        topic = (intent.get("topic") or "").strip()
        if not topic:
            await message.answer("Уточни тему для новостей.")
            return
        try:
            hours = int(intent.get("hours") or 24)
        except (TypeError, ValueError):
            hours = 24
        hours = max(1, min(168, hours))
        await message.answer(
            sanitize_html(f"📰 Готовлю дайджест: <i>{topic}</i> · окно {hours}ч…")
        )
        text = await build_news_digest(client, message.from_user.id, topic, hours=hours)
        await message.answer(sanitize_html(text), disable_web_page_preview=True)
        return

    # ниже — интенты, требующие конкретного контакта
    contact_query = (intent.get("contact") or "").strip()
    if not contact_query:
        await message.answer("🤷 Не понял, с каким контактом работать. Уточни имя.")
        return

    candidates = await resolve(client, owner, contact_query)
    if not candidates:
        await message.answer(
            sanitize_html(f"🙅 Не нашёл контакт «{contact_query}». Попробуй /sync.")
        )
        return

    action_map = {
        "summarize_chat": "summary",
        "tasks_for_chat": "tasks",
        "draft_reply": "draft",
        "catchup": "catchup",
    }
    cb_action = action_map.get(kind)
    if cb_action is None:
        await message.answer("❓ Неизвестное действие.")
        return

    if len(candidates) > 1 and candidates[0].score < 90:
        await message.answer(
            f"С кем именно? (действие: <b>{cb_action}</b>)",
            reply_markup=_candidates_keyboard_chat(cb_action, candidates),
        )
        return

    target = candidates[0]
    ctx_store.set_last_peer(message.from_user.id, target.peer_id, target.display_name)
    await message.answer(f"⏳ Подгружаю чат с <b>{target.label()}</b>…")
    messages_loaded = await load_chat(
        client,
        message.from_user.id,
        target.peer_id,
        limit=CHAT_LOAD_LIMIT,
        transcribe=True,
    )
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        contact = await get_contact(session, owner, target.peer_id)
        provider = await build_provider(session, owner)
        heavy = owner.settings.use_heavy_model

    if contact is None or provider is None:
        await message.answer("⚠️ Не удалось подготовить контекст.")
        return

    if kind == "summarize_chat":
        text = await summarize_chat(
            provider,
            contact,
            messages_loaded,
            heavy=heavy,
            global_style=owner.global_style_profile,
            owner_id=owner.id,
        )
        await message.answer(
            sanitize_html(f"📝 <b>Саммари — {contact.display_name}</b>\n\n{text}")
        )

    elif kind == "tasks_for_chat":
        items = await extract_and_save_commitments(
            provider,
            telegram_id=owner.telegram_id,
            contact_name=contact.display_name,
            contact_peer_id=contact.peer_id,
            messages=messages_loaded,
        )
        if not items:
            body = "🤷 Явных обязательств не нашёл."
        else:
            lines = []
            for it in items:
                who = "Я" if it.get("direction") == "mine" else "Они"
                deadline = it.get("deadline")
                tail = f" · до {deadline}" if deadline else ""
                lines.append(f"• <b>{who}</b>: {it.get('text', '')}{tail}")
            body = "\n".join(lines)
        await message.answer(
            sanitize_html(f"✅ <b>Обязательства — {contact.display_name}</b>\n\n{body}")
        )

    elif kind == "draft_reply":
        instruction = intent.get("instruction") or None
        draft = await draft_reply(
            provider,
            contact,
            messages_loaded,
            instruction=instruction,
            heavy=heavy,
            global_style=owner.global_style_profile,
            owner_id=owner.id,
        )
        payload = json.dumps(
            {"peer_id": target.peer_id, "text": draft}, ensure_ascii=False
        )
        async with get_session() as session:
            owner = await get_or_create_user(session, message.from_user.id)
            from src.db.repo import create_pending_action

            action = await create_pending_action(
                session, user_id=owner.id, kind="send_message", payload=payload
            )
        await message.answer(
            sanitize_html(
                f"💬 <b>Черновик — {contact.display_name}</b>\n\n{draft}\n\nОтправить?"
            ),
            reply_markup=_confirm_keyboard(action.id),
        )

    elif kind == "catchup":
        text = await catchup(
            provider,
            contact,
            messages_loaded,
            heavy=heavy,
            global_style=owner.global_style_profile,
            owner_id=owner.id,
        )
        await message.answer(
            sanitize_html(
                f"⏪ <b>Где мы остановились — {contact.display_name}</b>\n\n{text}"
            )
        )


async def _find_chats_and_offer(message, client, query: str, action: str) -> None:
    from src.core.contacts.chat_finder import smart_find

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        provider = await build_provider(session, owner)
    if provider is None:
        await message.answer("Нужен LLM-ключ (/settings → 🔑).")
        return

    try:
        results = await smart_find(client, owner, provider, query, top_n=5)
    except Exception:
        logger.exception("smart_find failed")
        await message.answer("❌ Поиск не удался. Попробуй ещё раз или уточни запрос.")
        return

    if not results:
        await message.answer(
            sanitize_html(
                f"Ничего не нашёл по «{query}» — ни по тексту, ни по именам контактов. "
                "Попробуй описать чуть конкретнее или назови сам контакт."
            )
        )
        return

    # пишем контакты в БД, чтобы chat:<action>:<peer_id> handler знал display_name
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        for r in results:
            await upsert_contact(
                session,
                owner,
                peer_id=r.peer_id,
                peer_kind=r.kind,
                is_bot=r.is_bot,
                display_name=r.name,
                username=r.username,
            )

    kb = InlineKeyboardBuilder()
    for r in results:
        marks = []
        if r.text_hits:
            marks.append(f"{r.text_hits} совп.")
        if r.name_score:
            marks.append(f"имя {r.name_score}/5")
        meta = " · ".join(marks)
        label = f"{r.name}" + (f" · {meta}" if meta else "")
        if len(label) > 60:
            label = label[:57] + "…"
        kb.row(
            InlineKeyboardButton(text=label, callback_data=f"chat:{action}:{r.peer_id}")
        )
    kb.row(InlineKeyboardButton(text="❌ Отмена", callback_data="chat:cancel:0"))

    pretty_action = {
        "catchup": "«где остановились»",
        "summary": "саммари",
        "tasks": "задачи/обещания",
        "draft": "черновик ответа",
    }.get(action, action)

    await message.answer(
        f"Нашёл подходящие чаты. Выбери — соберу {pretty_action}:",
        reply_markup=kb.as_markup(),
    )


async def _process_text(
    raw: str,
    message: Message,
    state: FSMContext,
    userbot_manager: UserbotManager,
) -> None:
    turn_started = time.monotonic()
    used_skills: list[dict] = []
    ctx = await _get_owner_context(message.from_user.id)
    tz_name = str(ctx["tz_name"])
    owner_telegram_id = int(ctx["owner_telegram_id"])  # type: ignore[arg-type]
    use_heavy = bool(ctx["use_heavy"])

    now_local_str = now_in_tz(tz_name).strftime("%Y-%m-%d %H:%M")
    history_block = ctx_store.render_history_block(message.from_user.id)

    # Adaptive instructions — проверяем не инструкция ли это
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
                    await apply_instruction(owner_telegram_id, instr["rule"])
                    await message.answer(
                        sanitize_html(f"✅ Понял! Больше не буду {instr['rule']}.")
                    )
                    await session.flush()
                    return
                else:
                    candidate = InstructionCandidate(
                        user_id=owner_db.id,
                        rule=instr["rule"],
                        category=instr["category"],
                        is_safe=False,
                        llm_reviewed=False,  # будет дообработан InstructionOptimizer
                    )
                    session.add(candidate)
                    await session.flush()
                    await message.answer(
                        sanitize_html(
                            f"🤔 Понял: «{instr['rule']}». Применить это правило? (да/нет)"
                        )
                    )
                    return
    except Exception:
        logger.exception("adaptive instruction check failed")

    # Adaptive persona — авто-подстройка стиля
    try:
        from src.core.intelligence.adaptive_persona import (
            detect_persona_change,
            apply_persona_changes,
        )

        change = await detect_persona_change(raw)
        if change:
            await apply_persona_changes(owner_telegram_id, change["changes"])
            await message.answer(sanitize_html(f"✅ Понял! Буду {change['reason']}."))
            return
    except Exception:
        logger.exception("adaptive persona check failed")

    # Follow-up контекст: проверяем, не продолжение ли это предыдущего запроса
    followup = _detect_followup(raw, owner_telegram_id)
    if followup:
        intent, update_type = followup
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
        return

    # Smart AutoRouter — оркестрация
    _last_purpose = None
    try:
        _last_purpose = ctx_store.get_last_purpose(message.from_user.id)
    except Exception:
        logger.exception("failed to get last purpose")
    router_plan = await make_plan(
        raw,
        owner_telegram_id,
        heavy_available=use_heavy,
        last_purpose=_last_purpose,
    )
    if router_plan.tasks:
        t0 = router_plan.tasks[0]
        logger.debug(
            "AutoRouter plan: risk=%s purpose=%s heavy=%s cache_ttl=%d agents=%s",
            t0.risk.value,
            t0.purpose.value,
            t0.heavy,
            t0.cache_ttl,
            t0.need_agents or "—",
        )

    # INSTANT — отвечаем сразу, без БД, без LLM
    if router_plan.response_mode == "instant" and router_plan.final_response:
        await message.answer(sanitize_html(router_plan.final_response))
        _fire_record_trajectory(
            owner_telegram_id,
            request_text=raw,
            route_mode="instant",
            intent_json={"intent": "chat"},
            response_text=router_plan.final_response,
            success=True,
            latency_ms=int((time.monotonic() - turn_started) * 1000),
        )
        await _post_turn_optimize(owner_telegram_id, raw, router_plan.final_response)
        return

    # Строим провайдер с учётом purpose из auto-router'а
    purpose = router_plan.tasks[0].purpose.value if router_plan.tasks else "main"
    async with get_session() as session:
        owner_db = await get_or_create_user(session, owner_telegram_id)
        provider = await build_provider(session, owner_db, purpose=purpose)
        if provider is None and purpose != "main":
            logger.debug("No key for purpose '%s', falling back to main", purpose)
            provider = await build_provider(session, owner_db, purpose="main")

    if provider is None:
        await message.answer(
            "Чтобы я мог понимать свободный текст — добавь LLM-ключ в /settings → 🔑 API-ключи."
        )
        return

    # FAST_ROUTE — один light LLM-вызов с готовым контекстом
    if router_plan.response_mode == "fast_route":
        # Собираем контекст через prompt_assembler (Block 4)
        fast_system = "Ты ассистент. Ответь коротко."
        try:
            from src.core.intelligence.prompt_assembler import (
                AssemblyContext,
                prompt_assembler,
            )

            persona_block = ""
            try:
                from src.core.intelligence.adaptive_persona import (
                    format_persona_for_prompt,
                )

                persona_block = await format_persona_for_prompt(owner_telegram_id) or ""
            except Exception:
                logger.debug("fast_route persona load skipped", exc_info=True)
                pass

            fast_ctx = AssemblyContext(
                target="summarizer",  # лёгкий режим — без тяжёлых блоков
                user_id=owner_telegram_id,
                memory_context=router_plan.memory_context or "",
                self_profile=router_plan.self_profile or "",
                persona_block=persona_block,
            )
            fast_ctx.skill_index, used_skills = await build_skill_index(
                owner_telegram_id, raw, "fast_route"
            )
            fast_system = prompt_assembler.assemble(fast_ctx)
        except Exception:
            # Fallback для fast route
            fast_context_parts = []
            if router_plan.memory_context:
                fast_context_parts.append(router_plan.memory_context)
            if router_plan.self_profile:
                fast_context_parts.append(router_plan.self_profile)
            try:
                from src.core.intelligence.adaptive_persona import (
                    format_persona_for_prompt,
                )

                persona_hint = await format_persona_for_prompt(owner_telegram_id)
                if persona_hint:
                    fast_context_parts.append(persona_hint)
            except Exception:
                logger.debug("fast_route persona load skipped", exc_info=True)
                pass
            fast_system = (
                "Ты ассистент. Ответь коротко.\n\n" + "\n\n".join(fast_context_parts)
                if fast_context_parts
                else "Ты ассистент. Ответь коротко."
            )
        fast_start = time.monotonic()
        try:
            from src.llm.base import ChatMessage

            fast_reply = await provider.chat(
                [
                    ChatMessage(role="system", content=fast_system),
                    ChatMessage(role="user", content=raw),
                ],
                heavy=False,
            )
            router_plan.final_response = fast_reply
            router_plan.metrics["llm_ms"] = int((time.monotonic() - fast_start) * 1000)
        except Exception as e:
            logger.warning("Fast route failed: %s", e)
            router_plan.metrics["llm_ms"] = -1
        else:
            router_plan.metrics["total_ms"] = router_plan.metrics.get(
                "recall_ms", 0
            ) + router_plan.metrics.get("llm_ms", 0)
            logger.info(
                "Fast route metrics: %s", json.dumps(router_plan.metrics, default=str)
            )
            await message.answer(sanitize_html(router_plan.final_response))

            # Траектория + скиллы — в фон, не блокируем ответ
            async def _save_trajectory_bg():
                try:
                    tid = await record_trajectory(
                        owner_telegram_id,
                        request_text=raw,
                        route_mode="fast_route",
                        intent_json={"intent": "chat"},
                        used_skills_json=used_skills,
                        response_text=router_plan.final_response,
                        success=True,
                        latency_ms=int((time.monotonic() - turn_started) * 1000),
                    )
                    await record_skill_usages(owner_telegram_id, used_skills, tid, True)
                except Exception:
                    logger.debug("fast_route bg trajectory save failed", exc_info=True)

            asyncio.create_task(_save_trajectory_bg())
            await _post_turn_optimize(
                owner_telegram_id, raw, router_plan.final_response
            )
            return

    # MAESTRO — стандартный тяжёлый пайплайн
    # Persona инжектится через prompt_assembler в maestro.py (system prompt, не дублируем)
    injected_style = ctx.get("global_style_profile") or None
    rag_needed = router_plan.recall_mode == "deep"
    try:
        pipeline_result = await run_pipeline(
            provider,
            raw,
            owner_id=owner_telegram_id,
            history_block=history_block,
            memory_context=router_plan.memory_context,
            global_style=injected_style,
            rag_enabled=rag_needed,
        )
        response_text = pipeline_result.get("final_response", "")
        if response_text:
            # Логгируем что сработало
            used = pipeline_result.get("used_agents", [])
            errors = pipeline_result.get("agent_errors", [])
            if used:
                logger.debug("Maestro agents: %s", used)
            if errors:
                logger.debug("Maestro agent errors: %s", errors)
            await message.answer(
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
            await _post_turn_optimize(owner_telegram_id, raw, response_text)
            return
    except Exception:
        logger.debug("Maestro pipeline failed, falling back to route_intent")

    try:
        intent = await route_intent(
            provider,
            raw,
            heavy=False,
            now_local=now_local_str,
            tz_name=tz_name,
            history_block=history_block,
            memory_context=router_plan.memory_context,
            user_id=owner_telegram_id,
        )
    except Exception as e:
        logger.exception("agent route_intent failed")
        err_msg = str(e)
        _fire_record_trajectory(
            owner_telegram_id,
            request_text=raw,
            route_mode="intent",
            success=False,
            error=err_msg[:4000],
            latency_ms=int((time.monotonic() - turn_started) * 1000),
        )
        if len(err_msg) > 300:
            err_msg = err_msg[:300] + "…"
        await message.answer(
            f"❌ Ошибка при обработке запроса.\n\n"
            f"<code>{err_msg}</code>\n\n"
            "<i>Если ошибка повторяется — проверь ключ в /settings → 🔑 API-ключи "
            "и модель в /settings → 🤖 LLM.</i>"
        )
        return

    if intent.get("intent") == "multi":
        actions = intent.get("actions") or []
        if not isinstance(actions, list) or not actions:
            await message.answer("Не понял, что сделать.")
            return
        for sub in actions:
            await _dispatch(sub, message, state, userbot_manager, tz_name=tz_name)
    elif "intents" in intent:
        # Multi-intent array: dispatch each sub-intent sequentially
        for sub in intent["intents"]:
            await _dispatch(sub, message, state, userbot_manager, tz_name=tz_name)
    else:
        await _dispatch(intent, message, state, userbot_manager, tz_name=tz_name)

    _save_intent_context(owner_telegram_id, intent)

    _fire_record_trajectory(
        owner_telegram_id,
        request_text=raw,
        route_mode="intent",
        intent_json=intent,
        actions_json=actions_from_intent(intent),
        response_text=_summarize_intent_for_memory(intent),
        success=True,
        latency_ms=int((time.monotonic() - turn_started) * 1000),
    )

    summary = _summarize_intent_for_memory(intent)
    ctx_store.add_turn(message.from_user.id, raw, summary)
    # Обновляем last_purpose для context chaining
    try:
        if router_plan and router_plan.tasks:
            ctx_store.set_last_purpose(
                message.from_user.id, router_plan.tasks[0].purpose.value
            )
    except Exception:
        logger.exception("failed to set last purpose")


@router.message(F.text & ~F.text.startswith("/"))
async def free_text(
    message: Message,
    state: FSMContext,
    userbot_manager: UserbotManager,
) -> None:
    if await state.get_state() is not None:
        return
    raw = (message.text or "").strip()
    if not raw:
        return
    if len(raw) > 2000:
        raw = raw[:1997] + "...(truncated)"
    await _process_text(raw, message, state, userbot_manager)


@router.message(F.voice | F.audio)
async def free_voice(
    message: Message,
    state: FSMContext,
    userbot_manager: UserbotManager,
) -> None:
    if await state.get_state() is not None:
        return

    media = message.voice or message.audio
    if media is None:
        return

    # 1. Быстрая загрузка настроек пользователя из БД
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        mode = owner.settings.transcription_mode
        openai_key = await get_api_key(session, owner, "openai")
        gemini_key = await get_api_key(session, owner, "gemini")
        mistral_key = await get_api_key(session, owner, "mistral")
        api_provider = getattr(owner.settings, "transcription_api_provider", "openai")

    # 2. Скачивание .ogg файла (быстрая сетевая операция)
    media_dir = settings.data_dir / "media" / "control_bot"
    media_dir.mkdir(parents=True, exist_ok=True)
    target = media_dir / f"{message.message_id}_{media.file_unique_id}.ogg"

    try:
        await message.bot.download(media.file_id, destination=str(target))
    except Exception:
        logger.exception("voice download failed")
        await message.answer("❌ Не удалось скачать голосовое.")
        return

    # 3. Ставим в очередь фоновой обработки (транскрипция + process_text)
    await _voice_queue.put(
        (
            target,
            message,
            state,
            userbot_manager,
            media.file_unique_id,
            mode,
            api_provider,
            openai_key,
            gemini_key,
            mistral_key,
        )
    )

    # 4. Мгновенный ответ — пользователь не ждёт транскрипцию
    await message.answer("🎙 Принял, расшифровываю…")


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
        await message.answer(f"⚠️ Действие остановлено guardrail: {guard.reason}")
        return
    intent = guard.intent
    kind = intent.get("intent")
    if kind == "set_setting":
        await _exec_set_setting(intent, message)
        return
    if kind == "add_news_topic":
        await _exec_add_news_topic(intent, message)
        return
    if kind == "remove_news_topic":
        await _exec_remove_news_topic(intent, message)
        return
    if kind == "add_reminder":
        await _exec_add_reminder(intent, message, tz_name=tz_name)
        return
    if kind == "remove_reminder":
        await _exec_remove_reminder(intent, message)
        return
    if kind == "add_reminders_from_chat":
        await _exec_add_reminders_from_chat(intent, message, userbot_manager)
        return
    if kind == "store_memory":
        await _exec_store_memory(intent, message)
        return
    if kind == "forget_memory":
        await _exec_forget_memory(intent, message)
        return
    if kind == "list_memories":
        await _exec_list_memories(intent, message)
        return
    if kind == "extract_memories_from_chat":
        await _exec_extract_memories(intent, message, userbot_manager)
        return
    if kind == "check_memories":
        await _exec_check_memories(intent, message)
        return
    if kind == "change_auto_mode":
        await _exec_change_auto_mode(intent, message)
        return
    if kind == "set_quiet_hours":
        await _exec_set_quiet_hours(intent, message)
        return
    if kind == "show_inbox":
        await _exec_show_inbox(intent, message, userbot_manager)
        return
    if kind == "show_self":
        await _exec_show_self(intent, message)
        return
    if kind == "full_analysis":
        await _exec_full_analysis(intent, message)
        return
    if kind == "add_api_key":
        await _exec_add_api_key(intent, message)
        return
    if kind == "remove_api_key":
        await _exec_remove_api_key(intent, message)
        return
    if kind == "toggle_api_key":
        await _exec_toggle_api_key(intent, message)
        return
    if kind == "list_keys":
        await _exec_list_keys(intent, message)
        return
    if kind == "show_digest":
        await _exec_show_digest(intent, message)
        return
    if kind == "show_today":
        await _exec_show_today(intent, message)
        return
    if kind == "show_skills":
        await _exec_show_skills(intent, message)
        return
    if kind == "show_threads":
        await _exec_show_threads(intent, message)
        return
    if kind == "show_trajectory":
        await _exec_show_trajectory(intent, message)
        return
    if kind == "show_style":
        await _exec_show_style(intent, message)
        return
    if kind == "show_profile":
        await _exec_show_profile(intent, message)
        return
    if kind == "index_chats":
        await _exec_index_chats(intent, message)
        return
    if kind == "clarify":
        question = (intent.get("question") or "").strip()
        if question:
            await message.answer(sanitize_html(f"🤔 {question}"))
        else:
            await message.answer("Не совсем понял. Уточни, что имеешь в виду?")
        return
    await _execute_intent(intent, message, state, userbot_manager, tz_name=tz_name)


async def _exec_add_reminder(intent, message, *, tz_name: str) -> None:
    text = (intent.get("text") or "").strip()
    if not text:
        await message.answer("🤷 Не понял, о чём напомнить. Уточни.")
        return
    when = _parse_iso_to_utc_naive(intent.get("when"), tz_name)
    peer_query = (intent.get("peer_query") or "").strip()

    peer_id = 0
    peer_name = None
    if peer_query:
        client = get_active_telethon_client(message.from_user.id)
        if client is not None:
            async with get_session() as session:
                owner = await get_or_create_user(session, message.from_user.id)
            cands = await resolve(client, owner, peer_query)
            if cands:
                peer_id = cands[0].peer_id
                peer_name = cands[0].display_name

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        await add_commitment(
            session,
            user_id=owner.id,
            peer_id=peer_id,
            peer_name=peer_name,
            message_id=None,
            direction="mine",
            text=text,
            deadline_at=when,
        )
        reminders_enabled = owner.settings.reminders_enabled if owner.settings else True

    when_str = fmt_local(when, tz_name) if when else "без срока"
    extra = f" (контакт: {peer_name})" if peer_name else ""
    note = (
        ""
        if reminders_enabled
        else "\n\n⚠ Напоминания выключены — включи в /settings → ⏰."
    )
    await message.answer(
        sanitize_html(
            f"⏰ Напоминание добавлено: <b>{text}</b>\nКогда: {when_str}{extra}{note}"
        )
    )


async def _exec_remove_reminder(intent, message) -> None:
    needle = (intent.get("query") or "").strip().lower()
    if not needle:
        await message.answer("Какое напоминание убрать?")
        return
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        items = await list_open_commitments(session, owner)
        matched = [
            c
            for c in items
            if needle in (c.text or "").lower()
            or (c.peer_name and needle in c.peer_name.lower())
        ]
        if not matched:
            await message.answer(
                sanitize_html(f"🙅 Не нашёл напоминаний по «{needle}».")
            )
            return
        for c in matched:
            await update_commitment_status(session, c.id, "cancelled")
    names = "\n".join(f"• {c.text}" for c in matched)
    await message.answer(f"🗑 Снял ({len(matched)}):\n{names}")


async def _exec_add_reminders_from_chat(intent, message, userbot_manager) -> None:
    contact_query = (intent.get("contact") or "").strip()
    if not contact_query:
        await message.answer("С каким контактом извлечь обещания?")
        return
    client = userbot_manager.get_client(message.from_user.id)
    if client is None:
        await message.answer("Сначала /login.")
        return

    from src.core.contacts.contact_resolver import resolve

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        provider = await build_provider(session, owner)
    if provider is None:
        await message.answer("Нужен LLM-ключ (/settings → 🔑).")
        return

    cands = await resolve(client, owner, contact_query)
    if not cands:
        await message.answer(sanitize_html(f"Контакт «{contact_query}» не найден."))
        return
    target = cands[0]

    await message.answer(
        f"⏳ Подгружаю чат с <b>{target.label()}</b> и извлекаю обещания…"
    )
    msgs = await load_chat(
        client, message.from_user.id, target.peer_id, limit=80, transcribe=True
    )
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        contact = await get_contact(session, owner, target.peer_id)
        owner_telegram_id = owner.telegram_id
        contact_id = contact.id
        contact_display = contact.display_name
    items = await extract_and_save_commitments(
        provider,
        telegram_id=owner_telegram_id,
        contact_name=contact_display,
        contact_peer_id=contact_id,
        messages=msgs,
    )
    if not items:
        await message.answer("🤷 Явных обещаний в этом чате не нашёл.")
        return
    lines = []
    for it in items:
        who = "Я" if it.get("direction") == "mine" else "Они"
        deadline = it.get("deadline")
        tail = f" · до {deadline}" if deadline else ""
        lines.append(f"• <b>{who}</b>: {it.get('text', '')}{tail}")
    await message.answer(
        sanitize_html(
            f"⏰ Поставил {len(items)} напоминаний из чата с {target.display_name}:\n\n"
            + "\n".join(lines)
        )
    )


async def _exec_show_inbox(intent, message, userbot_manager) -> None:
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        from src.db.repo import list_active_conversations

        conversations = await list_active_conversations(session, owner, status="active")
        waiting = await list_active_conversations(
            session, owner, status="waiting_reply"
        )

    if not conversations and not waiting:
        await message.answer("📭 Нет активных переписок.")
        return

    lines = ["📬 <b>Входящие:</b>", ""]
    if conversations:
        lines.append(f"🟢 Активные ({len(conversations)}):")
        for c in conversations[:10]:
            async with get_session() as session:
                owner = await get_or_create_user(session, message.from_user.id)
                contact = await get_contact(session, owner, c.peer_id)
            name = contact.display_name if contact else str(c.peer_id)
            unread = f" ({c.unread_count})" if c.unread_count > 1 else ""
            lines.append(f"  • {name}{unread}")

    if waiting:
        lines.append(f"🟡 Ждут ответа ({len(waiting)}):")
        for c in waiting[:10]:
            async with get_session() as session:
                owner = await get_or_create_user(session, message.from_user.id)
                contact = await get_contact(session, owner, c.peer_id)
            name = contact.display_name if contact else str(c.peer_id)
            lines.append(f"  • {name}")

    await message.answer("\n".join(lines))


async def _exec_show_self(intent: dict, message: Message) -> None:
    """Показать что бот знает о пользователе (self-profile + recall + fuel)."""
    from src.core.memory.memory_fuel import get_fuel_stats, format_depleted_contacts
    from src.core.memory.memory_recall import recall, format_recall_human

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        prof = await get_self_profile(session, owner)
        owner_telegram_id = owner.telegram_id

    lines = ["🧑 <b>Что я знаю о тебе:</b>", ""]

    if prof:
        if prof.preferences:
            lines.append(f"❤️ Предпочтения: {prof.preferences}")
        if prof.goals:
            lines.append(f"🎯 Цели: {prof.goals}")
        if prof.current_projects:
            lines.append(f"📂 Проекты: {prof.current_projects}")
        if prof.decision_style:
            lines.append(f"🤔 Стиль решений: {prof.decision_style}")
        if prof.sleep_pattern:
            lines.append(f"😴 Сон: {prof.sleep_pattern}")
        if prof.work_hours:
            lines.append(f"💼 Работа: {prof.work_hours}")

    # Recall (self-факты)
    try:
        result = await recall(
            owner_telegram_id,
            limit=5,
            include_self=True,
            include_pinned=True,
            include_tasks=False,
        )
        if result.facts:
            lines.append("")
            lines.append("🧠 <b>Что помню:</b>")
            lines.append(format_recall_human(result))
    except Exception:
        logger.exception("self recall failed")

    # Чего НЕ знаю (fuel gauge — истощённые зоны)
    try:
        fuel = await get_fuel_stats(owner_telegram_id)
        if fuel.get("depleted"):
            lines.append("")
            lines.append("🤷 <b>Чего НЕ знаю:</b>")
            lines.append(format_depleted_contacts(fuel))
    except Exception:
        logger.exception("fuel stats failed")

    text = "\n".join(lines)
    await message.answer(text)


async def _exec_full_analysis(intent, message) -> None:
    folders = intent.get("folders") or []
    await message.answer(
        sanitize_html(
            f"🧠 Запускаю полный анализ{' папок: ' + ', '.join(folders) if folders else ' всех контактов'}..."
        )
    )
    status_msg = await message.answer("⏳ Подготовка...")

    async def _run():
        async with get_session() as session:
            owner = await get_or_create_user(session, message.from_user.id)
            provider = await build_provider(session, owner)
            if not provider:
                await status_msg.edit_text("❌ Нет LLM провайдера.")
                return
        from src.core.infra.full_analyzer import (
            run_full_analysis,
            format_analysis_report,
        )

        result = await run_full_analysis(
            owner_id=message.from_user.id,
            provider=provider,
            message_limit=500,
            folder_names=folders if folders else None,
        )
        report = format_analysis_report(result)
        await status_msg.edit_text(sanitize_html(report))

    import asyncio

    asyncio.create_task(_run())


# ─── Key management handlers (natural language) ─────────────────────


async def _exec_add_api_key(intent: dict, message: Message) -> None:
    """Добавить API-ключ(и) через естественный язык."""
    provider = (intent.get("provider") or "").strip().lower()
    purpose = (intent.get("purpose") or "main").strip().lower()
    keys_raw = (intent.get("key") or "").strip()

    if not provider or not keys_raw:
        await message.answer(
            "🤷 Не хватает данных: укажи провайдера (openai/gemini/mistral) и ключ."
        )
        return

    if provider not in ("openai", "gemini", "mistral"):
        await message.answer("❌ Провайдер: openai, gemini или mistral")
        return

    # Split by comma for bulk
    keys = [k.strip() for k in keys_raw.split(",") if k.strip()]
    if not keys:
        await message.answer("❌ Пустой список ключей.")
        return

    # Delete message with keys from chat (security)
    try:
        await message.delete()
    except Exception:
        logger.exception("failed to delete message with key")

    from src.crypto import decrypt
    from src.llm.router import _provider_class_for

    success = 0
    failed = 0
    results = []
    last_slot_id = None

    for i, api_key in enumerate(keys):
        async with get_session() as session:
            owner = await get_or_create_user(session, message.from_user.id)
            slot = await add_key_slot(
                session,
                owner,
                provider,
                api_key,
                purpose=purpose,
                label=f"{provider}/{purpose}",
                priority=i,
            )
            last_slot_id = slot.id
        # Validate key
        try:
            raw_key = decrypt(slot.key_enc)
            prov_class = _provider_class_for(provider)
            prov = prov_class(raw_key)
            valid = await prov.validate_key()
            if not valid:
                async with get_session() as session:
                    owner = await get_or_create_user(session, message.from_user.id)
                    bad_slot = await session.get(LlmKeySlot, slot.id)
                    if bad_slot:
                        await session.delete(bad_slot)
                        await session.flush()
                results.append(f"  #{slot.id} {api_key[:12]}… ❌")
                failed += 1
            else:
                results.append(f"  #{slot.id} {api_key[:12]}… ✅")
                success += 1
        except Exception:
            results.append(f"  #{slot.id} {api_key[:12]}… ✅ (ошибка проверки)")
            success += 1

    if len(keys) == 1 and success == 1:
        await message.answer(
            f"✅ Ключ {provider}/{purpose} добавлен и проверен! (слот #{last_slot_id})"
        )
    elif len(keys) == 1 and failed == 1:
        await message.answer(
            f"❌ Ключ {provider}/{purpose} не прошёл валидацию. Проверь ключ."
        )
    else:
        lines = [f"<b>Добавлено {len(keys)} ключей {provider}/{purpose}:</b>", ""]
        lines.extend(results)
        await message.answer("\n".join(lines))


# ─── NL intent: show_* and index_chats ────────────────────────────────


async def _exec_show_digest(intent: dict, message: Message) -> None:
    """Показать утренний дайджест (build_digest)."""
    from src.core.scheduling.digest import build_digest

    text = await build_digest(message.from_user.id)
    await message.answer(sanitize_html(text) if text else "Дайджест пуст.")


async def _exec_show_today(intent: dict, message: Message) -> None:
    """Показать сводку за сегодня (smart_digest)."""
    from src.core.scheduling.smart_digest import (
        build_smart_digest,
        collect_recent_messages,
    )

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        interval = owner.settings.smart_digest_interval_min
        messages = await collect_recent_messages(session, owner, since_minutes=interval)
        text = build_smart_digest(messages, interval)
    await message.answer(sanitize_html(text) if text else "На сегодня ничего.")


async def _exec_show_skills(intent: dict, message: Message) -> None:
    """Показать список навыков."""
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        from src.db.repo import list_skills

        skills = await list_skills(session, owner)
    if skills:
        text = "📊 <b>Навыки:</b>\n" + "\n".join(f"• {s.name}" for s in skills)
    else:
        text = "Навыков пока нет."
    await message.answer(sanitize_html(text))


async def _exec_show_threads(intent: dict, message: Message) -> None:
    """Показать активные треды (conversation states)."""
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        from src.db.repo import list_active_conversations

        convs = await list_active_conversations(session, owner)
    if convs:
        lines = []
        for c in convs[:10]:
            async with get_session() as session:
                owner = await get_or_create_user(session, message.from_user.id)
                from src.db.repo import get_contact

                contact = await get_contact(session, owner, c.peer_id)
            name = contact.display_name if contact else str(c.peer_id)
            unread = c.unread_count or 0
            lines.append(f"• {name} (непрочитано: {unread})")
        text = "🧵 <b>Активные треды:</b>\n" + "\n".join(lines)
    else:
        text = "Активных тредов нет."
    await message.answer(sanitize_html(text))


async def _exec_show_trajectory(intent: dict, message: Message) -> None:
    """Показать траекторию действий."""
    only_errors = intent.get("only_errors", False)
    limit = intent.get("limit", 10)
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        from src.db.repo import list_trajectories

        items = await list_trajectories(
            session, owner, only_errors=only_errors, limit=limit
        )
    if items:
        lines = []
        for t in items:
            action = (t.intent_json or {}).get("intent", "?") if t.intent_json else "?"
            success = "✅" if t.success else "❌"
            lines.append(f"{success} {action}")
        text = "📜 <b>Траектория:</b>\n" + "\n".join(lines)
    else:
        text = "Траектория пуста."
    await message.answer(sanitize_html(text))


async def _exec_show_style(intent: dict, message: Message) -> None:
    """Показать стиль общения (контактный или глобальный)."""
    from src.core.contacts.style_profile import update_style_profile_for_contact

    contact_name = (intent.get("contact_name") or "").strip()
    if contact_name:
        async with get_session() as session:
            owner = await get_or_create_user(session, message.from_user.id)
            from src.db.repo import list_contacts

            contacts = await list_contacts(
                session, owner, kinds=("user", "group"), include_bots=False
            )
            matched = [
                c
                for c in contacts
                if contact_name.lower() in (c.display_name or "").lower()
            ]
            if matched:
                target = matched[0]
                provider = await build_provider(session, owner)
                if provider:
                    profile = await update_style_profile_for_contact(
                        provider, message.from_user.id, target.peer_id
                    )
                    text = sanitize_html(str(profile))
                else:
                    text = "Не удалось создать LLM провайдер."
            else:
                text = f"Контакт '{contact_name}' не найден."
        await message.answer(text)
    else:
        async with get_session() as session:
            owner = await get_or_create_user(session, message.from_user.id)
            gs = owner.global_style_profile
            text = (
                sanitize_html(str(gs))
                if gs
                else "Глобальный стиль не собран. /style для сбора."
            )
        await message.answer(text)


async def _exec_show_profile(intent: dict, message: Message) -> None:
    """Показать профиль пользователя (перенаправление на show_self)."""
    await _exec_show_self(intent, message)


async def _exec_index_chats(intent: dict, message: Message) -> None:
    """Переиндексация чатов — перенаправление на /index."""
    await message.answer("Для переиндексации используй команду /index")


async def _exec_remove_api_key(intent: dict, message: Message) -> None:
    """Удалить слот ключа через естественный язык."""
    slot_id = intent.get("slot_id")
    remove_all = intent.get("all")

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)

        # Remove all
        if remove_all and str(remove_all).strip().lower() in ("все", "all"):
            slots = await list_key_slots(session, owner)
            if not slots:
                await message.answer("❌ Нет ключей для удаления.")
                return
            count = 0
            for s in slots:
                await session.delete(s)
                count += 1
            await session.commit()
            await message.answer(f"✅ Удалено {count} ключей.")
            return

        if slot_id is None:
            await message.answer("🤷 Не указан номер слота. Напиши: удали ключ 5")
            return

        try:
            slot_id = int(slot_id)
        except (TypeError, ValueError):
            await message.answer("❌ Номер слота должен быть числом.")
            return

        slot = await session.get(LlmKeySlot, slot_id)
        if slot and slot.user_id == owner.id:
            prov = slot.provider
            purp = slot.purpose
            await session.delete(slot)
            await session.commit()
            await message.answer(f"✅ Слот #{slot_id} ({prov}/{purp}) удалён.")
        else:
            await message.answer("❌ Слот не найден или не твой.")


async def _exec_toggle_api_key(intent: dict, message: Message) -> None:
    """Включить/выключить слот ключа через естественный язык."""
    slot_id = intent.get("slot_id")
    action = (intent.get("action") or "toggle").strip().lower()

    if slot_id is None:
        await message.answer("🤷 Не указан номер слота. Напиши: отключи ключ 3")
        return

    try:
        slot_id = int(slot_id)
    except (TypeError, ValueError):
        await message.answer("❌ Номер слота должен быть числом.")
        return

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        slot = await session.get(LlmKeySlot, slot_id)
        if slot and slot.user_id == owner.id:
            if action == "enable":
                slot.enabled = True
            elif action == "disable":
                slot.enabled = False
            else:  # toggle
                slot.enabled = not slot.enabled
            await session.commit()
            status = "включён" if slot.enabled else "выключен"
            await message.answer(
                f"✅ Слот #{slot_id} ({slot.provider}/{slot.purpose}) {status}."
            )
        else:
            await message.answer("❌ Слот не найден или не твой.")


async def _exec_list_keys(intent: dict, message: Message) -> None:
    """Показать все ключи через естественный язык."""
    from datetime import datetime, timezone

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        slots = await list_key_slots(session, owner)

    if not slots:
        await message.answer(
            "🔑 <b>Нет ключевых слотов.</b>\n\n"
            "Добавь ключ через /keys add openai main sk-...\n"
            "Где:\n"
            "• провайдер: openai/gemini/mistral\n"
            "• purpose: main/draft/memory/background/search/analysis/urgent/fallback\n"
            "• ключ: сам API ключ"
        )
        return

    lines = ["<b>🔑 Ключевые слоты:</b>", ""]
    for s in slots[:10]:
        status = "✅" if s.enabled else "🚫"
        cool = (
            " 🔒"
            if (cooldown := _ensure_utc(s.cooldown_until))
            and cooldown > datetime.now(timezone.utc)
            else ""
        )
        lines.append(
            f"{status} <b>{s.provider}</b> / {s.purpose} "
            f"(приоритет {s.priority}, исп. {s.usage_count}×{cool})"
        )
        if s.last_error:
            lines.append(f"   ⚠️ {s.last_error[:80]}")
        if s.label:
            lines.append(f"   🏷 {s.label}")
    await message.answer("\n".join(lines))
