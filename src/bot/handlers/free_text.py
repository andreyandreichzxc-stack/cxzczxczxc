"""Свободный текст (и голос) → агент → действие. Регистрируется последним в bot/app.py,
чтобы команды и FSM перехватывали свои события раньше."""

import json
import logging
from pathlib import Path

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
from src.config import settings as app_settings
from src.core.agent import route_intent
from src.core.chat_service import load_chat
from src.core.maestro import run_pipeline
from src.core.commitment_extractor import extract_and_save_commitments
from src.core.contact_resolver import resolve, resolve_with_llm
from src.core.news import build_news_digest
from src.core.summarizer import catchup, draft_reply, summarize_chat
from src.core import conversation_context as ctx_store
from src.core.text_sanitizer import sanitize_html
from src.core.timeutil import fmt_local, is_valid_tz, now_in_tz, tz_short
from src.core.transcription import transcription_service
from src.db.repo import (
    add_commitment,
    add_memory,
    add_memory_candidate,
    add_news_topic,
    create_pending_action,
    delete_memory,
    delete_news_topic,
    get_api_key,
    get_contact,
    get_contact_profile,
    get_memory_stats,
    get_or_create_user,
    get_self_profile,
    list_memories,
    list_news_topics,
    list_open_commitments,
    search_memories,
    update_commitment_status,
    upsert_contact,
)
from src.db.session import get_session
from src.llm.router import build_provider
from src.userbot.manager import UserbotManager, _MANAGER_SINGLETON


logger = logging.getLogger(__name__)
router = Router(name="free_text")
router.message.filter(OwnerOnly())


CHAT_LOAD_LIMIT = 50


# Поля UserSettings, которые агент может менять через set_setting (имя → тип значения)
SETTING_FIELDS: dict[str, str] = {
    "auto_reply_enabled": "bool",
    "auto_reply_mode": "choice:static,smart",
    "auto_reply_text": "str",
    "auto_reply_cooldown_min": "int",
    "digest_enabled": "bool",
    "digest_time": "hm",
    "news_enabled": "bool",
    "news_digest_time": "hm",
    "news_window_hours": "int",
    "reminders_enabled": "bool",
    "reminder_lead_hours": "int",
    "reminder_overdue_enabled": "bool",
    "ignore_archived": "bool",
    "use_heavy_model": "bool",
    "llm_provider": "choice:openai,gemini,mistral",
    "transcription_mode": "choice:local,api,hybrid",
    "transcription_api_provider": "choice:openai,gemini,mistral",
    "auto_sync_enabled": "bool",
    "auto_sync_interval_sec": "int",
    "auto_extract_memories": "bool",
    "include_saved_messages": "bool",
    "smart_digest_enabled": "bool",
    "smart_digest_interval_min": "int",
    "urgent_notify_enabled": "bool",
    "monitor_only_selected_folders": "bool",
    "monitored_folders": "str",
    "timezone": "tz",
}


import re

_HM_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


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
        if isinstance(raw, str) and _HM_RE.match(raw.strip()):
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
        from src.core.timeutil import fmt_local

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
            await message.answer(f"Не нашёл контакт «{recipient}». Попробуй /sync.")
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
                action = await create_pending_action(
                    session, user_id=owner.id, kind="send_message", payload=payload
                )
                # подгружаем факты о собеседнике
                facts_hint = ""
                profile_hint = ""
                neg_warning = ""
                if target.peer_id:
                    contact_facts: list = []
                    try:
                        from src.core.memory_recall import recall, format_recall_human

                        contact_facts_result = await recall(
                            owner.telegram_id,
                            contact_id=target.peer_id,
                            query=text[:200],
                            limit=3,
                            include_self=False,
                            include_pinned=False,
                            include_tasks=False,
                        )
                        if contact_facts_result.facts:
                            fact_lines = [
                                f"{rf.reason}: {rf.fact[:60]}"
                                for rf in contact_facts_result.facts
                            ]
                            facts_hint = "\n\n📝 О собеседнике: " + "; ".join(
                                fact_lines
                            )
                    except Exception:
                        pass

                    # подгружаем профиль (стиль, dos/donts)
                    try:
                        profile = await get_contact_profile(
                            session, owner, target.peer_id
                        )
                        if profile:
                            hints = []
                            if profile.communication_style:
                                hints.append(profile.communication_style)
                            if profile.communication_dos:
                                import json as _json

                                dos = (
                                    _json.loads(profile.communication_dos)
                                    if profile.communication_dos.startswith("[")
                                    else [profile.communication_dos]
                                )
                                hints.append(f"✅ {', '.join(dos[:3])}")
                            if profile.communication_donts:
                                import json as _json

                                donts = (
                                    _json.loads(profile.communication_donts)
                                    if profile.communication_donts.startswith("[")
                                    else [profile.communication_donts]
                                )
                                hints.append(f"❌ {', '.join(donts[:3])}")
                            if hints:
                                profile_hint = "\n\n👤 Профиль: " + " | ".join(hints)
                    except Exception:
                        pass

                    # Негативные факты о контакте — предупреждение
                    try:
                        from datetime import datetime, timezone

                        neg_mems = await list_memories(
                            session, owner, contact_id=target.peer_id
                        )
                        recent_neg = [
                            m
                            for m in neg_mems
                            if m.sentiment == "negative"
                            and m.created_at
                            and (datetime.now(timezone.utc) - m.created_at).days < 7
                        ]
                        if recent_neg:
                            neg_warning = (
                                "\n\n⚠️ <b>Внимание:</b> за последнюю неделю негативные факты: "
                                + "; ".join(m.fact[:50] for m in recent_neg[:2])
                            )
                    except Exception:
                        pass

            await message.answer(
                f"🤔 <b>Готов отправить</b>\n\n"
                f"→ <b>Кому:</b> {target.label()}\n"
                f"→ <b>Текст:</b>\n{text}{facts_hint}{neg_warning}{profile_hint}",
                reply_markup=_confirm_keyboard(action.id),
            )
        else:
            await state.set_data({"send_text": text})
            await message.answer(
                f"Кому именно отправить «<i>{text[:80]}</i>»?",
                reply_markup=_candidates_keyboard_send(candidates),
            )
        return

    if kind == "search":
        query = (intent.get("query") or "").strip()
        peer_query = (intent.get("peer_query") or intent.get("contact") or "").strip()
        if not query:
            await message.answer("Не понял, что искать.")
            return
        await message.answer(f"🔎 Ищу: <i>{query}</i>…")
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
        await message.answer(f"🔎 Ищу по моим чатам: «<i>{query}</i>»…")
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
        await message.answer(f"📰 Готовлю дайджест: <i>{topic}</i> · окно {hours}ч…")
        text = await build_news_digest(client, message.from_user.id, topic, hours=hours)
        await message.answer(text, disable_web_page_preview=True)
        return

    # ниже — интенты, требующие конкретного контакта
    contact_query = (intent.get("contact") or "").strip()
    if not contact_query:
        await message.answer("🤷 Не понял, с каким контактом работать. Уточни имя.")
        return

    candidates = await resolve(client, owner, contact_query)
    if not candidates:
        await message.answer(f"🙅 Не нашёл контакт «{contact_query}». Попробуй /sync.")
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
        await message.answer(f"📝 <b>Саммари — {contact.display_name}</b>\n\n{text}")

    elif kind == "tasks_for_chat":
        items = await extract_and_save_commitments(
            provider,
            telegram_id=owner.telegram_id,
            contact=contact,
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
            f"✅ <b>Обязательства — {contact.display_name}</b>\n\n{body}"
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
            action = await create_pending_action(
                session, user_id=owner.id, kind="send_message", payload=payload
            )
        await message.answer(
            f"💬 <b>Черновик — {contact.display_name}</b>\n\n{draft}\n\nОтправить?",
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
            f"⏪ <b>Где мы остановились — {contact.display_name}</b>\n\n{text}"
        )


async def _find_chats_and_offer(message, client, query: str, action: str) -> None:
    from src.core.chat_finder import smart_find

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
            f"Ничего не нашёл по «{query}» — ни по тексту, ни по именам контактов. "
            "Попробуй описать чуть конкретнее или назови сам контакт."
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


async def _exec_set_setting(intent, message) -> None:
    key = (intent.get("key") or "").strip()
    value = intent.get("value")
    spec = SETTING_FIELDS.get(key)
    if spec is None:
        await message.answer(f"Не умею менять «{key}».")
        return
    validated, err = _coerce_setting_value(spec, value)
    if err:
        await message.answer(f"Не понял значение для <b>{key}</b>: {err}.")
        return
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        setattr(owner.settings, key, validated)
        new_tz = owner.settings.timezone
    if key == "timezone":
        await message.answer(f"✅ Часовой пояс: <b>{tz_short(new_tz)}</b>")
    elif isinstance(validated, bool):
        await message.answer(f"✅ <b>{key}</b>: {'ВКЛ' if validated else 'ВЫКЛ'}")
    else:
        shown = str(validated)
        if len(shown) > 100:
            shown = shown[:97] + "…"
        await message.answer(f"✅ <b>{key}</b> = <code>{shown}</code>")


async def _exec_add_news_topic(intent, message) -> None:
    topic = (intent.get("topic") or "").strip()
    if not topic:
        await message.answer("Не понял какую тему добавить.")
        return
    try:
        hours = int(intent.get("hours") or 24)
    except (TypeError, ValueError):
        hours = 24
    hours = max(1, min(168, hours))
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        await add_news_topic(session, owner, topic, hours=hours)
    await message.answer(f"✅ Добавил тему: <b>{topic}</b> (окно {hours}ч)")


async def _exec_remove_news_topic(intent, message) -> None:
    needle = (intent.get("topic") or "").strip().lower()
    if not needle:
        await message.answer("Какую тему удалить?")
        return
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        topics = await list_news_topics(session, owner)
        matched = [t for t in topics if needle in t.topic.lower()]
        if not matched:
            await message.answer(f"Тем по «{needle}» не нашёл.")
            return
        for t in matched:
            await delete_news_topic(session, owner, t.id)
    names = ", ".join(f"«{t.topic}»" for t in matched)
    await message.answer(f"🗑 Удалил: {names}")


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


async def _process_text(
    raw: str,
    message: Message,
    state: FSMContext,
    userbot_manager: UserbotManager,
) -> None:
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        provider = await build_provider(session, owner)
        tz_name = owner.settings.timezone

    if provider is None:
        await message.answer(
            "Чтобы я мог понимать свободный текст — добавь LLM-ключ в /settings → 🔑 API-ключи."
        )
        return

    now_local_str = now_in_tz(tz_name).strftime("%Y-%m-%d %H:%M")
    history_block = ctx_store.render_history_block(message.from_user.id)

    # грузим память для контекста через recall
    memory_context = ""
    try:
        from src.core.memory_recall import recall, format_recall_for_prompt

        recall_result = await recall(
            owner.telegram_id,
            query=raw[:200],
            limit=12,
            include_self=True,
            include_pinned=True,
            include_tasks=True,
        )
        memory_context = format_recall_for_prompt(recall_result)
    except Exception:
        pass

    # Сначала пробуем Maestro + агенты — полный пайплайн
    try:
        pipeline_result = await run_pipeline(
            provider,
            raw,
            owner_id=owner.telegram_id,
            history_block=history_block,
            memory_context=memory_context,
            global_style=getattr(owner, "global_style_profile", None),
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
                response_text,
                reply_markup=memory_quick_keyboard(),
            )
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
            memory_context=memory_context,
            user_id=owner.telegram_id,
        )
    except Exception as e:
        logger.exception("agent route_intent failed")
        err_msg = str(e)
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
    else:
        await _dispatch(intent, message, state, userbot_manager, tz_name=tz_name)

    summary = _summarize_intent_for_memory(intent)
    ctx_store.add_turn(message.from_user.id, raw, summary)


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

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        mode = owner.settings.transcription_mode
        openai_key = await get_api_key(session, owner, "openai")
        gemini_key = await get_api_key(session, owner, "gemini")
        mistral_key = await get_api_key(session, owner, "mistral")
        api_provider = getattr(owner.settings, "transcription_api_provider", "openai")

    media_dir = app_settings.data_dir / "media" / "control_bot"
    media_dir.mkdir(parents=True, exist_ok=True)
    target = media_dir / f"{message.message_id}_{media.file_unique_id}.ogg"

    notice = await message.answer("🎙 Слушаю… (транскрибирую)")
    try:
        await message.bot.download(media.file_id, destination=str(target))
        text = await transcription_service.transcribe(
            target,
            file_id=media.file_unique_id,
            mode=mode,
            openai_key=openai_key,
            gemini_key=gemini_key,
            mistral_key=mistral_key,
            api_provider=api_provider,
        )
    except Exception:
        logger.exception("voice transcription failed")
        try:
            await notice.edit_text("❌ Не удалось распознать голосовое.")
        except Exception:
            pass
        return

    text = (text or "").strip()
    if not text:
        try:
            await notice.edit_text("Не услышал текста в этом сообщении.")
        except Exception:
            pass
        return

    try:
        await notice.edit_text(f"🎙 <i>Услышал:</i> {text}")
    except Exception:
        pass

    await _process_text(text, message, state, userbot_manager)


async def _dispatch(intent, message, state, userbot_manager, *, tz_name: str) -> None:
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
    if kind == "clarify":
        question = (intent.get("question") or "").strip()
        if question:
            await message.answer(f"🤔 {question}")
        else:
            await message.answer("Не совсем понял. Уточни, что имеешь в виду?")
        return
    await _execute_intent(intent, message, state, userbot_manager, tz_name=tz_name)


def _parse_iso_to_utc_naive(value, tz_name: str | None = None):
    if not value:
        return None
    try:
        from datetime import datetime, timezone
        from src.core.timeutil import parse_tz

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
        from src.userbot.manager import _MANAGER_SINGLETON

        client = (
            _MANAGER_SINGLETON.get_client(message.from_user.id)
            if _MANAGER_SINGLETON
            else None
        )
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

    when_str = fmt_local(when, tz_name) if when else "без срока"
    extra = f" (контакт: {peer_name})" if peer_name else ""
    note = (
        ""
        if owner.settings.reminders_enabled
        else "\n\n⚠ Напоминания выключены — включи в /settings → ⏰."
    )
    await message.answer(
        f"⏰ Напоминание добавлено: <b>{text}</b>\nКогда: {when_str}{extra}{note}"
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
            await message.answer(f"🙅 Не нашёл напоминаний по «{needle}».")
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

    from src.core.contact_resolver import resolve

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        provider = await build_provider(session, owner)
    if provider is None:
        await message.answer("Нужен LLM-ключ (/settings → 🔑).")
        return

    cands = await resolve(client, owner, contact_query)
    if not cands:
        await message.answer(f"Контакт «{contact_query}» не найден.")
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
    items = await extract_and_save_commitments(
        provider, telegram_id=owner.telegram_id, contact=contact, messages=msgs
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
        f"⏰ Поставил {len(items)} напоминаний из чата с {target.display_name}:\n\n"
        + "\n".join(lines)
    )


async def _exec_store_memory(intent, message) -> None:
    fact = (intent.get("fact") or "").strip()
    if not fact:
        await message.answer("🤷 Не понял, что запомнить. Уточни.")
        return
    contact_name = (intent.get("contact") or "").strip()
    sentiment = (intent.get("sentiment") or "").strip()
    if sentiment not in ("positive", "negative", "neutral"):
        sentiment = None

    # Confidence из интента; если нет — считаем низкой (→ кандидат)
    confidence = float(intent.get("confidence") or 0.0)

    contact_id = None
    if contact_name:
        async with get_session() as session:
            owner = await get_or_create_user(session, message.from_user.id)
        client = (
            _MANAGER_SINGLETON.get_client(message.from_user.id)
            if _MANAGER_SINGLETON
            else None
        )
        if client is not None:
            candidates = await resolve(client, owner, contact_name)
            if candidates:
                contact_id = candidates[0].peer_id

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)

        if confidence >= 0.85:
            # Высокая уверенность — сразу в память
            mem = await add_memory(
                session,
                owner,
                fact=fact,
                contact_id=contact_id,
                sentiment=sentiment,
                source="user",
            )
            await message.answer(f"🧠 Запомнил: <i>{fact}</i>")
        else:
            # Низкая уверенность — в черновик (MemoryCandidate)
            await add_memory_candidate(
                session,
                owner,
                fact=fact,
                contact_id=contact_id,
                sentiment=sentiment,
                source="user",
            )
            await message.answer(
                f"📬 Сохранил как черновик: <i>{fact}</i>\n"
                f"Подтверди через <code>/memory --inbox</code>"
            )


async def _exec_forget_memory(intent, message) -> None:
    query = (intent.get("query") or "").strip()
    if not query:
        await message.answer("Что удалить? Уточни.")
        return
    contact_name = (intent.get("contact") or "").strip()

    contact_id = None
    if contact_name:
        async with get_session() as session:
            owner = await get_or_create_user(session, message.from_user.id)
        client = (
            _MANAGER_SINGLETON.get_client(message.from_user.id)
            if _MANAGER_SINGLETON
            else None
        )
        if client is not None:
            candidates = await resolve(client, owner, contact_name)
            if candidates:
                contact_id = candidates[0].peer_id

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        found = await search_memories(session, owner, query, contact_id=contact_id)

    if not found:
        await message.answer("Ничего не нашёл по этому запросу.")
        return

    async with get_session() as session:
        for m in found:
            await delete_memory(session, owner, m.id)

    names = ", ".join(
        f"«{m.fact[:50]}…»" if len(m.fact) > 50 else f"«{m.fact}»" for m in found
    )
    await message.answer(f"🗑 Забыл: {names}")


async def _exec_list_memories(intent, message) -> None:
    contact_name = (intent.get("contact") or "").strip()

    contact_id = None
    label = ""
    if contact_name:
        async with get_session() as session:
            owner = await get_or_create_user(session, message.from_user.id)
        client = (
            _MANAGER_SINGLETON.get_client(message.from_user.id)
            if _MANAGER_SINGLETON
            else None
        )
        if client is not None:
            candidates = await resolve(client, owner, contact_name)
            if candidates:
                contact_id = candidates[0].peer_id
                label = f" — {candidates[0].label()}"

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        items = await list_memories(session, owner, contact_id=contact_id)
        items = [m for m in items if m.is_active]

    if not items:
        await message.answer("Память пуста.")
        return

    lines = []
    for m in items:
        sent = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(
            m.sentiment or "", ""
        )
        lines.append(f"• {sent} {m.fact}")
    body = "\n".join(lines)
    await message.answer(f"🧠 <b>Память{label}</b>\n\n{body}")


async def _exec_extract_memories(intent, message, userbot_manager) -> None:
    contact_name = (intent.get("contact") or "").strip()
    if not contact_name:
        await message.answer("Про какой контакт извлечь память?")
        return

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)

    client = (
        userbot_manager.get_client(message.from_user.id) if userbot_manager else None
    )
    if client is None:
        await message.answer("Сначала /login.")
        return

    candidates = await resolve(client, owner, contact_name)
    if not candidates:
        await message.answer("Не нашёл такого контакта.")
        return

    peer_id = candidates[0].peer_id

    from src.core.chat_service import load_chat, message_to_text
    from src.core.memory_queue import enqueue, MemoryJob

    # Загружаем сообщения и строим транскрипт
    messages = await load_chat(client, message.from_user.id, peer_id, limit=100)
    transcript = "\n".join(message_to_text(m) for m in messages)

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        contact = await get_contact(session, owner, peer_id)

    # Ставим задачу в очередь на фоновое извлечение
    await enqueue(
        MemoryJob(
            telegram_id=message.from_user.id,
            contact_id=contact.peer_id if contact else None,
            messages_text=transcript,
            job_type="extract",
        )
    )
    await message.answer("🧠 Извлекаю факты в фоне…")


async def _exec_change_auto_mode(intent, message) -> None:
    mode = (intent.get("mode") or "").strip()
    if mode not in ("offline_only", "always", "smart"):
        await message.answer("❌ Укажи режим: offline_only, always или smart")
        return
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        owner.settings.auto_mode = mode
        await session.flush()
    labels = {"offline_only": "только оффлайн", "always": "всегда", "smart": "умный"}
    await message.answer(f"✅ Режим авто-ответа: <b>{labels[mode]}</b>")


async def _exec_set_quiet_hours(intent, message) -> None:
    start = (intent.get("start") or "").strip()
    end = (intent.get("end") or "").strip()
    if not _HM_RE.match(start) or not _HM_RE.match(end):
        await message.answer("❌ Укажи время в формате HH:MM (например 23:00 и 07:00)")
        return
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        owner.settings.quiet_hours_start = start
        owner.settings.quiet_hours_end = end
        await session.flush()
    await message.answer(f"✅ Тихие часы: <b>{start} – {end}</b>")


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
    from src.core.memory_fuel import get_fuel_stats, format_depleted_contacts
    from src.core.memory_recall import recall, format_recall_human

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        prof = await get_self_profile(session, owner)

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
            owner.telegram_id,
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
        pass

    # Чего НЕ знаю (fuel gauge — истощённые зоны)
    try:
        fuel = await get_fuel_stats(owner.telegram_id)
        if fuel.get("depleted"):
            lines.append("")
            lines.append("🤷 <b>Чего НЕ знаю:</b>")
            lines.append(format_depleted_contacts(fuel))
    except Exception:
        pass

    text = "\n".join(lines)
    await message.answer(text)


async def _exec_full_analysis(intent, message) -> None:
    folders = intent.get("folders") or []
    await message.answer(
        f"🧠 Запускаю полный анализ{' папок: ' + ', '.join(folders) if folders else ' всех контактов'}..."
    )
    status_msg = await message.answer("⏳ Подготовка...")

    async def _run():
        async with get_session() as session:
            owner = await get_or_create_user(session, message.from_user.id)
            provider = await build_provider(session, owner)
            if not provider:
                await status_msg.edit_text("❌ Нет LLM провайдера.")
                return
        from src.core.full_analyzer import run_full_analysis, format_analysis_report

        result = await run_full_analysis(
            owner_id=message.from_user.id,
            provider=provider,
            message_limit=500,
            folder_names=folders if folders else None,
        )
        report = format_analysis_report(result)
        await status_msg.edit_text(report)

    import asyncio

    asyncio.create_task(_run())


async def _exec_check_memories(intent, message) -> None:
    """Бот сам задаёт вопросы про устаревшие факты из памяти."""
    questions = intent.get("questions") or []
    if not isinstance(questions, list) or not questions:
        return

    for q in questions[:2]:  # не больше 2 вопросов за раз
        mid = q.get("memory_id")
        question = q.get("question", "")
        if not question:
            continue
        kb = InlineKeyboardBuilder()
        kb.row(
            InlineKeyboardButton(text="✅ Да, всё ок", callback_data=f"mem:ok:{mid}"),
            InlineKeyboardButton(
                text="❌ Уже неактуально", callback_data=f"mem:del:{mid}"
            ),
        )
        await message.answer(f"🤔 {question}", reply_markup=kb.as_markup())


@router.callback_query(F.data.startswith("mem:ok:"))
async def cb_mem_ok(callback: CallbackQuery) -> None:
    from src.db.repo import get_or_create_user, list_memories

    mid = int(callback.data.split(":")[2])
    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        memories = await list_memories(session, owner)
        for m in memories:
            if m.id == mid:
                m.sentiment = "neutral"
    if callback.message:
        await callback.message.edit_text(
            f"✅ {callback.message.text}\n\n<i>Понял, память обновлена.</i>"
        )
    await callback.answer()


@router.callback_query(F.data.startswith("mem:del:"))
async def cb_mem_del(callback: CallbackQuery) -> None:
    from src.db.repo import delete_memory, get_or_create_user

    mid = int(callback.data.split(":")[2])
    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        await delete_memory(session, owner, mid)
    if callback.message:
        await callback.message.edit_text(
            f"🗑 {callback.message.text}\n\n<i>Удалил из памяти.</i>"
        )
    await callback.answer()


# ── Memory Quick Actions (inline-кнопки) ──────────────────────────────


@router.callback_query(F.data == "memq:list")
async def cb_memq_list(callback: CallbackQuery) -> None:
    """Показать последние 10 фактов памяти."""
    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        memories = await list_memories(session, owner)
        active = [m for m in memories if m.is_active]
        if not active:
            await callback.answer("Память пуста 📭", show_alert=True)
            return
        lines = ["<b>🧠 Последние факты:</b>", ""]
        for m in active[:10]:
            emoji = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(
                m.sentiment, "⚪"
            )
            lines.append(f"{emoji} {m.fact[:100]}")
        lines.append(f"\n<i>Всего: {len(memories)} фактов. /memory — подробнее</i>")
        await callback.message.answer("\n".join(lines))
        await callback.answer()


@router.callback_query(F.data == "memq:add")
async def cb_memq_add(callback: CallbackQuery) -> None:
    """Предложить добавить факт в память."""
    await callback.message.answer(
        "📝 <b>Что запомнить?</b>\n"
        "Напиши факт в формате:\n"
        "<code>запомни: [факт]</code>\n\n"
        "Например: <code>запомни: у Насти ДР 15 июня</code>"
    )
    await callback.answer()


@router.callback_query(F.data == "memq:forget")
async def cb_memq_forget(callback: CallbackQuery) -> None:
    """Показать последние факты для удаления."""
    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        memories = await list_memories(session, owner)
        active = [m for m in memories if m.is_active]
        if not active:
            await callback.answer("Нечего забывать 📭", show_alert=True)
            return
        kb = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text=f"❌ {m.fact[:40]}", callback_data=f"memq:del:{m.id}"
                    )
                ]
                for m in active[:8]
            ]
        )
        await callback.message.answer(
            "<b>❌ Что забыть?</b>\nВыбери факт для удаления:",
            reply_markup=kb,
        )
        await callback.answer()


@router.callback_query(F.data.startswith("memq:del:"))
async def cb_memq_delete(callback: CallbackQuery) -> None:
    """Удалить конкретный факт памяти по ID."""
    mem_id = int(callback.data.split(":")[2])
    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        success = await delete_memory(session, owner, mem_id)
        if success:
            await callback.message.edit_text("✅ Забыто!")
        else:
            await callback.answer("Не удалось удалить", show_alert=True)
    await callback.answer()


@router.callback_query(F.data.startswith("memq:explain:"))
async def cb_memq_explain(callback: CallbackQuery) -> None:
    """Показать объяснение (почему бот так думает)."""
    contact_name = callback.data.split(":", 2)[2] if ":" in callback.data else ""

    contact_id = None
    contact_label = ""
    if contact_name:
        # Пытаемся найти контакт
        from src.userbot.manager import _MANAGER_SINGLETON

        client = (
            _MANAGER_SINGLETON.get_client(callback.from_user.id)
            if _MANAGER_SINGLETON
            else None
        )
        if client is not None:
            async with get_session() as session:
                owner = await get_or_create_user(session, callback.from_user.id)
            from src.core.contact_resolver import resolve

            candidates = await resolve(client, owner, contact_name)
            if candidates:
                contact_id = candidates[0].peer_id
                contact_label = candidates[0].label()

    from src.bot.handlers.explain_cmd import build_explain_text

    text = await build_explain_text(
        callback.from_user.id,
        contact_id=contact_id,
        contact_label=contact_label,
    )
    if callback.message:
        await callback.message.answer(text)
    await callback.answer()
