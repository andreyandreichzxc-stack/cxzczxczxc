"""Intent exec handlers extracted from free_text.py, free_text_settings.py, free_text_memory.py.
Each function handles a specific intent kind.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

from aiogram.filters import CommandObject
from aiogram.types import (
    InlineKeyboardButton,
    Message,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder

from src.core.actions.commitment_extractor import extract_and_save_commitments
from src.core.contacts.chat_service import load_chat
from src.core.contacts.contact_resolver import resolve, resolve_with_llm
from src.core.infra.text_sanitizer import sanitize_html
from src.core.infra.timeutil import (
    fmt_local,
)
from src.core.services.chat_actions import (
    catchup_action,
    draft_reply_action,
    extract_tasks_action,
    summarize_chat_action,
)
from src.core.memory import conversation_context as ctx_store
from src.core.scheduling.news import build_news_digest
from src.crypto import decrypt
from src.db.models import LlmKeySlot
from src.db.repo import (
    add_commitment,
    add_key_slot,
    get_contact,
    get_or_create_user,
    get_self_profile,
    list_key_slots,
    list_open_commitments,
    update_commitment_status,
    upsert_contact,
)
from src.db.session import get_session
from src.llm.router import _ensure_utc, build_provider, _provider_class_for
from src.userbot import get_active_telethon_client

from .free_text_common import (
    _candidates_keyboard_chat,
    _candidates_keyboard_send,
    _confirm_keyboard,
    _parse_iso_to_utc_naive,
)
from .free_text_memory import (
    _exec_store_memory,
    _exec_forget_memory,
    _exec_list_memories,
    _exec_extract_memories,
    _exec_check_memories,
)
from .free_text_settings import (
    _exec_set_setting,
    _exec_add_news_topic,
    _exec_remove_news_topic,
    _exec_change_auto_mode,
    _exec_set_quiet_hours,
)

# Re-export submodule functions with public names
exec_set_setting = _exec_set_setting
exec_add_news_topic = _exec_add_news_topic
exec_remove_news_topic = _exec_remove_news_topic
exec_change_auto_mode = _exec_change_auto_mode
exec_set_quiet_hours = _exec_set_quiet_hours
exec_store_memory = _exec_store_memory
exec_forget_memory = _exec_forget_memory
exec_list_memories = _exec_list_memories
exec_extract_memories = _exec_extract_memories
exec_check_memories = _exec_check_memories

logger = logging.getLogger(__name__)

CHAT_LOAD_LIMIT = 50


# ── Classic helper ───────────────────────────────────────────────────


async def classic_resolve_contact(
    intent: dict, message: Message, userbot_manager
) -> int | None:
    """Разрешить контакт для классических хендлеров. Возвращает peer_id или None."""
    contact_query = (intent.get("contact") or "").strip()
    if not contact_query:
        await message.answer("🤷 Не понял, с каким контактом работать. Уточни имя.")
        return None
    client = userbot_manager.get_client(message.from_user.id)
    if client is None:
        await message.answer("Сначала /login — нужен подключённый Telegram-аккаунт.")
        return None
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
    candidates = await resolve(client, owner, contact_query)
    if not candidates:
        await message.answer(
            sanitize_html(f"🙅 Не нашёл контакт «{contact_query}». Попробуй /sync.")
        )
        return None
    kind = intent.get("intent")
    action_map = {
        "summarize_chat": "summary",
        "tasks_for_chat": "tasks",
        "draft_reply": "draft",
        "catchup": "catchup",
    }
    cb_action = action_map.get(kind)
    if cb_action is None:
        await message.answer("❓ Неизвестное действие.")
        return None
    if len(candidates) > 1 and candidates[0].score < 90:
        await message.answer(
            f"С кем именно? (действие: <b>{cb_action}</b>)",
            reply_markup=_candidates_keyboard_chat(cb_action, candidates),
        )
        return None
    target = candidates[0]
    ctx_store.set_last_peer(message.from_user.id, target.peer_id, target.display_name)
    return target.peer_id


# ── Chat finder helper ───────────────────────────────────────────────


async def find_chats_and_offer(message, client, query: str, action: str) -> None:
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


# ── _exec_* handlers ─────────────────────────────────────────────────


async def exec_add_reminder(intent, message, *, tz_name: str) -> None:
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


async def exec_remove_reminder(intent, message) -> None:
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
    await message.answer(sanitize_html(f"🗑 Снял ({len(matched)}):\n{names}"))


async def exec_add_reminders_from_chat(intent, message, userbot_manager) -> None:
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
        contact_display = contact.display_name
    items = await extract_and_save_commitments(
        provider,
        telegram_id=owner_telegram_id,
        contact_name=contact_display,
        contact_peer_id=target.peer_id,
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


async def exec_show_inbox(intent, message, userbot_manager) -> None:
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
                contact = await get_contact(session, owner, c.peer_id)
                name = contact.display_name if contact else str(c.peer_id)
                unread = f" ({c.unread_count})" if c.unread_count > 1 else ""
                lines.append(f"  • {name}{unread}")

        if waiting:
            lines.append(f"🟡 Ждут ответа ({len(waiting)}):")
            for c in waiting[:10]:
                contact = await get_contact(session, owner, c.peer_id)
                name = contact.display_name if contact else str(c.peer_id)
                lines.append(f"  • {name}")

    await message.answer("\n".join(lines))


async def exec_show_self(intent: dict, message: Message) -> None:
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


async def exec_full_analysis(intent, message) -> None:
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

    asyncio.create_task(_run())


# ─── Key management handlers (natural language) ─────────────────────


async def exec_add_api_key(intent: dict, message: Message) -> None:
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

    success = 0
    failed = 0
    results = []
    last_slot_id = None

    for i, api_key in enumerate(keys):
        async with get_session() as session:
            owner = await get_or_create_user(session, message.from_user.id)
            slot, is_new = await add_key_slot(
                session,
                owner,
                provider,
                api_key,
                purpose=purpose,
                label=f"{provider}/{purpose}",
                priority=i,
            )
            last_slot_id = slot.id
            if not is_new:
                results.append(
                    f"  #{slot.id} {provider}/{purpose} — Этот ключ уже добавлен (слот #{slot.id})"
                )
                success += 1
                continue
        # Validate key (только для новых слотов)
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
                results.append(f"  #{slot.id} {provider}/{purpose} ❌")
                failed += 1
            else:
                results.append(f"  #{slot.id} {provider}/{purpose} ✅")
                success += 1
        except Exception:
            results.append(f"  #{slot.id} {provider}/{purpose} ❌ (ошибка проверки)")
            failed += 1

    if len(keys) == 1 and success == 1:
        if not is_new:
            await message.answer(
                f"ℹ️ Ключ {provider}/{purpose} уже был добавлен ранее (слот #{last_slot_id})."
            )
        else:
            await message.answer(
                f"✅ Ключ {provider}/{purpose} добавлен и проверен! (слот #{last_slot_id})"
            )
        return
    elif len(keys) == 1 and failed == 1:
        await message.answer(
            f"❌ Ключ {provider}/{purpose} не прошёл валидацию. Проверь ключ."
        )
        return
    else:
        lines = [f"<b>Добавлено {len(keys)} ключей {provider}/{purpose}:</b>", ""]
        lines.extend(results)
    await message.answer("\n".join(lines))


# ─── NL intent: show_* and index_chats ────────────────────────────────


async def exec_show_digest(intent: dict, message: Message) -> None:
    """Показать утренний дайджест (build_digest)."""
    from src.core.scheduling.digest import build_digest

    text = await build_digest(message.from_user.id)
    await message.answer(sanitize_html(text) if text else "Дайджест пуст.")


async def exec_show_today(intent: dict, message: Message) -> None:
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


async def exec_show_skills(intent: dict, message: Message) -> None:
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


async def exec_show_threads(intent: dict, message: Message) -> None:
    """Показать активные треды (conversation states)."""
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        from src.db.repo import list_active_conversations

        convs = await list_active_conversations(session, owner)
    if convs:
        lines = []
        async with get_session() as session:
            owner = await get_or_create_user(session, message.from_user.id)
            for c in convs[:10]:
                from src.db.repo import get_contact

                contact = await get_contact(session, owner, c.peer_id)
                name = contact.display_name if contact else str(c.peer_id)
                unread = c.unread_count or 0
                lines.append(f"• {name} (непрочитано: {unread})")
        text = "🧵 <b>Активные треды:</b>\n" + "\n".join(lines)
    else:
        text = "Активных тредов нет."
    await message.answer(sanitize_html(text))


async def exec_show_trajectory(intent: dict, message: Message) -> None:
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


async def exec_show_style(intent: dict, message: Message) -> None:
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


async def exec_show_profile(intent: dict, message: Message) -> None:
    """Показать профиль пользователя (перенаправление на show_self)."""
    await exec_show_self(intent, message)


async def exec_index_chats(intent: dict, message: Message) -> None:
    """Переиндексация чатов — перенаправление на /index."""
    await message.answer("Для переиндексации используй команду /index")


async def exec_remove_api_key(intent: dict, message: Message) -> None:
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


async def exec_toggle_api_key(intent: dict, message: Message) -> None:
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


async def exec_list_keys(intent: dict, message: Message) -> None:
    """Показать все ключи через естественный язык."""

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


# ─── Classic intent handlers (extracted from _execute_intent) ──────


async def exec_classic_chat(
    intent: dict, message: Message, state, userbot_manager, *, tz_name: str
) -> None:
    reply = sanitize_html(intent.get("reply"))
    if not reply:
        reply = "Готов помочь. Уточни, пожалуйста."
    await message.answer(reply)


async def exec_classic_unknown(
    intent: dict, message: Message, state, userbot_manager, *, tz_name: str
) -> None:
    await message.answer(
        "🤷 Не понял, что нужно сделать. Я умею: писать сообщения людям, делать саммари переписок, "
        "извлекать задачи, ловить «где мы остановились», искать по сообщениям, собирать новостной "
        "дайджест по теме, показывать обещания. Попробуй сформулировать иначе или открой /help."
    )


async def exec_classic_list_todos(
    intent: dict, message: Message, state, userbot_manager, *, tz_name: str
) -> None:
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        items = await list_open_commitments(session, owner)
    if not items:
        await message.answer("🎉 Открытых обязательств нет")
        return
    from src.core.infra.timeutil import fmt_local

    lines = []
    for c in items[:30]:
        who = "Я" if c.direction == "mine" else (sanitize_html(c.peer_name or "Они"))
        d = fmt_local(c.deadline_at, tz_name)
        lines.append(f"• <b>{who}</b>: {sanitize_html(c.text)} (до {d})")
    await message.answer(
        sanitize_html(
            f"📋 Открытых обязательств: <b>{len(items)}</b>\n\n" + "\n".join(lines)
        )
    )


async def exec_classic_send_message(
    intent: dict, message: Message, state, userbot_manager, *, tz_name: str
) -> None:
    client = userbot_manager.get_client(message.from_user.id)
    if client is None:
        await message.answer("Сначала /login — нужен подключённый Telegram-аккаунт.")
        return

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        provider = await build_provider(session, owner)

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
        if state is None:
            logger.warning(
                "FSMContext is None (stale/background) — skipping set_data for send_message"
            )
        else:
            await state.set_data({"send_text": text})
        await message.answer(
            sanitize_html(f"Кому именно отправить «<i>{text[:80]}</i>»?"),
            reply_markup=_candidates_keyboard_send(candidates),
        )


async def exec_classic_search(
    intent: dict, message: Message, state, userbot_manager, *, tz_name: str
) -> None:
    client = userbot_manager.get_client(message.from_user.id)
    if client is None:
        await message.answer("Сначала /login — нужен подключённый Telegram-аккаунт.")
        return

    query = (intent.get("query") or "").strip()
    if not query:
        await message.answer("Не понял, что искать.")
        return
    await message.answer(sanitize_html(f"🔎 Ищу: <i>{query}</i>…"))
    from src.bot.handlers.search import cmd_search

    await cmd_search(
        message,
        CommandObject(prefix="/", command="search", args=query),
        userbot_manager,
    )


async def exec_classic_find_in_chats(
    intent: dict, message: Message, state, userbot_manager, *, tz_name: str
) -> None:
    client = userbot_manager.get_client(message.from_user.id)
    if client is None:
        await message.answer("Сначала /login — нужен подключённый Telegram-аккаунт.")
        return

    query = (intent.get("query") or "").strip()
    action = (intent.get("action") or "catchup").strip()
    if action not in {"catchup", "summary", "tasks", "draft"}:
        action = "catchup"
    if not query:
        await message.answer("🤷 Не понял, по какой теме искать.")
        return
    await message.answer(sanitize_html(f"🔎 Ищу по моим чатам: «<i>{query}</i>»…"))
    await find_chats_and_offer(message, client, query, action)


async def exec_classic_news_digest(
    intent: dict, message: Message, state, userbot_manager, *, tz_name: str
) -> None:
    client = userbot_manager.get_client(message.from_user.id)
    if client is None:
        await message.answer("Сначала /login — нужен подключённый Telegram-аккаунт.")
        return

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


async def exec_classic_summarize_chat(
    intent: dict, message: Message, state, userbot_manager, *, tz_name: str
) -> None:
    peer_id = await classic_resolve_contact(intent, message, userbot_manager)
    if peer_id is None:
        return
    await message.answer("⏳ Подгружаю чат…")
    result = await summarize_chat_action(message.from_user.id, peer_id, userbot_manager)
    if result is None:
        await message.answer("⚠️ Не удалось подготовить контекст.")
        return
    await message.answer(sanitize_html(result.html), reply_markup=result.markup)


async def exec_classic_tasks_for_chat(
    intent: dict, message: Message, state, userbot_manager, *, tz_name: str
) -> None:
    peer_id = await classic_resolve_contact(intent, message, userbot_manager)
    if peer_id is None:
        return
    await message.answer("⏳ Подгружаю чат…")
    result = await extract_tasks_action(message.from_user.id, peer_id, userbot_manager)
    if result is None:
        await message.answer("⚠️ Не удалось подготовить контекст.")
        return
    await message.answer(sanitize_html(result.html), reply_markup=result.markup)


async def exec_classic_draft_reply(
    intent: dict, message: Message, state, userbot_manager, *, tz_name: str
) -> None:
    peer_id = await classic_resolve_contact(intent, message, userbot_manager)
    if peer_id is None:
        return
    await message.answer("⏳ Подгружаю чат…")
    instruction = (intent.get("instruction") or "").strip()
    result = await draft_reply_action(
        message.from_user.id,
        peer_id,
        userbot_manager,
        instruction=instruction,
    )
    if result is None:
        await message.answer("⚠️ Не удалось подготовить контекст.")
        return
    await message.answer(sanitize_html(result.html), reply_markup=result.markup)


async def exec_classic_catchup(
    intent: dict, message: Message, state, userbot_manager, *, tz_name: str
) -> None:
    peer_id = await classic_resolve_contact(intent, message, userbot_manager)
    if peer_id is None:
        return
    await message.answer("⏳ Подгружаю чат…")
    result = await catchup_action(message.from_user.id, peer_id, userbot_manager)
    if result is None:
        await message.answer("⚠️ Не удалось подготовить контекст.")
        return
    await message.answer(sanitize_html(result.html), reply_markup=result.markup)


# ─── Clarify handler ─────────────────────────────────────────────────


async def exec_clarify(intent: dict, message: Message) -> None:
    question = (intent.get("question") or "").strip()
    if question:
        await message.answer(sanitize_html(f"🤔 {question}"))
    else:
        await message.answer("Не совсем понял. Уточни, что имеешь в виду?")
