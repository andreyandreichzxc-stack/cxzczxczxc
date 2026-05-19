import asyncio
import json
import logging

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder

from src.bot.filters import OwnerOnly
from src.core.chat_service import load_chat
from src.core.commitment_extractor import extract_and_save_commitments
from src.core.memory_extractor import extract_and_save_memories
from src.core.contact_resolver import ContactCandidate, resolve
from src.core.summarizer import catchup, draft_reply, summarize_chat
from src.db.repo import get_contact, get_or_create_user, list_memories
from src.db.session import get_session
from src.llm.router import build_provider
from src.userbot.manager import UserbotManager
from src.bot.handlers.analyze_cmd import cmd_analyze


logger = logging.getLogger(__name__)
router = Router(name="chat_cmd")
router.message.filter(OwnerOnly())
router.callback_query.filter(OwnerOnly())


def _candidates_keyboard(
    action: str, candidates: list[ContactCandidate]
) -> InlineKeyboardMarkup:
    """Кнопки выбора контакта. callback_data: chat:<action>:<peer_id>"""
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


def _actions_keyboard(peer_id: int) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.row(
        InlineKeyboardButton(
            text="📝 Саммари", callback_data=f"chat:summary:{peer_id}"
        ),
        InlineKeyboardButton(
            text="✅ Задачи/обещания", callback_data=f"chat:tasks:{peer_id}"
        ),
    )
    kb.row(
        InlineKeyboardButton(
            text="💬 Черновик ответа", callback_data=f"chat:draft:{peer_id}"
        ),
        InlineKeyboardButton(
            text="⏪ Где мы остановились", callback_data=f"chat:catchup:{peer_id}"
        ),
    )
    kb.row(
        InlineKeyboardButton(text="📖 История", callback_data=f"chat:story:{peer_id}"),
        InlineKeyboardButton(text="🧠 Полный анализ", callback_data="chat:analyze:all"),
    )
    return kb.as_markup()


async def _ensure_client(message: Message, userbot_manager: UserbotManager):
    client = userbot_manager.get_client(message.from_user.id)
    if client is None:
        await message.answer("Сначала подключи аккаунт через /login.")
        return None
    return client


@router.message(Command("chat"))
async def cmd_chat(
    message: Message, command: CommandObject, userbot_manager: UserbotManager
) -> None:
    client = await _ensure_client(message, userbot_manager)
    if client is None:
        return

    query = (command.args or "").strip()
    if not query:
        # Показываем недавние контакты
        async with get_session() as session:
            owner = await get_or_create_user(session, message.from_user.id)
            from src.db.repo import fetch_chat_messages, list_contacts

            contacts = await list_contacts(
                session, owner, kinds=("user",), include_bots=False
            )
            # Берём последние сообщения для каждого контакта, сортируем по дате
            recent = []  # (display_name, peer_id, date)
            for ct in contacts[:30]:
                if ct.is_bot:
                    continue
                msgs = await fetch_chat_messages(session, owner, ct.peer_id, limit=1)
                if msgs:
                    recent.append((ct.display_name, ct.peer_id, msgs[0].date))

            recent.sort(key=lambda x: x[2], reverse=True)
            top5 = recent[:5]

        if not top5:
            await message.answer(
                "Использование: <code>/chat имя или @username</code>\n"
                "Пока нет недавних контактов. Попробуй /sync."
            )
            return

        lines = ["<b>💬 Недавние контакты</b>", "", "Выбери для действий:"]
        kb = InlineKeyboardBuilder()
        for name, pid, _date in top5:
            kb.row(
                InlineKeyboardButton(
                    text=f"💬 {name}",
                    callback_data=f"chat:pick:{pid}",
                )
            )
        kb.row(InlineKeyboardButton(text="❌ Отмена", callback_data="chat:cancel:0"))
        await message.answer(
            "\n".join(lines),
            reply_markup=kb.as_markup(),
        )
        return

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)

    candidates = await resolve(client, owner, query)
    if not candidates:
        await message.answer(
            "Не нашёл такого контакта. Уточни имя/ник или попробуй /sync."
        )
        return

    if len(candidates) == 1 or candidates[0].score >= 90:
        await _show_actions(message, candidates[0])
        return

    await message.answer(
        "Кого из них ты имел в виду?",
        reply_markup=_candidates_keyboard("pick", candidates),
    )


async def _show_actions(message: Message, candidate: ContactCandidate) -> None:
    memory_line = ""
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        memories = await list_memories(session, owner, contact_id=candidate.peer_id)
        if memories:
            facts_short = [m.fact[:40] for m in memories[:3]]
            memory_line = "🧠 Память: " + ", ".join(facts_short)
        else:
            memory_line = "🧠 Память: пока пусто"
    label = candidate.label()
    await message.answer(
        f"Выбран: <b>{label}</b>\n{memory_line}\n\nЧто сделать?",
        reply_markup=_actions_keyboard(candidate.peer_id),
    )


@router.callback_query(F.data.startswith("chat:cancel:"))
async def cb_cancel(callback: CallbackQuery) -> None:
    if callback.message:
        await callback.message.edit_text("Отменено.")
    await callback.answer()


@router.callback_query(F.data.startswith("chat:pick:"))
async def cb_pick(callback: CallbackQuery, userbot_manager: UserbotManager) -> None:
    peer_id = int(callback.data.split(":")[2])
    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        contact = await get_contact(session, owner, peer_id)
    label = contact.display_name if contact else str(peer_id)
    if callback.message:
        await callback.message.edit_text(
            f"Выбран: <b>{label}</b>. Что сделать?",
            reply_markup=_actions_keyboard(peer_id),
        )
    await callback.answer()


async def _action_load(
    callback: CallbackQuery, userbot_manager: UserbotManager, peer_id: int
):
    """Готовит данные для действий: client, owner, contact, messages, provider."""
    client = userbot_manager.get_client(callback.from_user.id)
    if client is None:
        await callback.answer("Подключи аккаунт через /login", show_alert=True)
        return None

    await callback.answer("Подгружаю чат…")
    if callback.message:
        await callback.message.edit_text("⏳ Подгружаю последние сообщения…")

    messages = await load_chat(
        client, callback.from_user.id, peer_id, limit=50, transcribe=True
    )
    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        contact = await get_contact(session, owner, peer_id)
        provider = await build_provider(session, owner)
        heavy = owner.settings.use_heavy_model

        # авто-извлечение памяти о контакте (fire-and-forget, не блокируем UI)
        import asyncio

        asyncio.create_task(
            extract_and_save_memories(provider, owner.id, contact, messages)
        )

    if contact is None:
        if callback.message:
            await callback.message.edit_text(
                "Контакт не найден в локальной БД. Попробуй /sync."
            )
        return None
    if provider is None:
        if callback.message:
            await callback.message.edit_text(
                "Не задан API-ключ выбранного LLM. Добавь в /settings."
            )
        return None
    return client, owner, contact, messages, provider, heavy


@router.callback_query(F.data.startswith("chat:summary:"))
async def cb_summary(callback: CallbackQuery, userbot_manager: UserbotManager) -> None:
    peer_id = int(callback.data.split(":")[2])
    bundle = await _action_load(callback, userbot_manager, peer_id)
    if bundle is None:
        return
    _client, _owner, contact, messages, provider, heavy = bundle

    text = await summarize_chat(
        provider, contact, messages, heavy=heavy, owner_id=_owner.id
    )
    if callback.message:
        await callback.message.edit_text(
            f"📝 <b>Саммари — {contact.display_name}</b>\n\n{text}",
            reply_markup=_actions_keyboard(peer_id),
        )


@router.callback_query(F.data.startswith("chat:tasks:"))
async def cb_tasks(callback: CallbackQuery, userbot_manager: UserbotManager) -> None:
    peer_id = int(callback.data.split(":")[2])
    bundle = await _action_load(callback, userbot_manager, peer_id)
    if bundle is None:
        return
    _client, owner, contact, messages, provider, _heavy = bundle

    items = await extract_and_save_commitments(
        provider,
        user_id=owner.id,
        contact=contact,
        messages=messages,
    )

    if not items:
        body = "Явных обязательств не нашёл."
    else:
        lines = []
        for it in items:
            who = "Я" if it.get("direction") == "mine" else "Они"
            deadline = it.get("deadline")
            tail = f" · до {deadline}" if deadline else ""
            lines.append(f"• <b>{who}</b>: {it.get('text', '')}{tail}")
        body = "\n".join(lines)

    if callback.message:
        await callback.message.edit_text(
            f"✅ <b>Обязательства — {contact.display_name}</b>\n\n{body}",
            reply_markup=_actions_keyboard(peer_id),
        )


@router.callback_query(F.data.startswith("chat:draft:"))
async def cb_draft(callback: CallbackQuery, userbot_manager: UserbotManager) -> None:
    peer_id = int(callback.data.split(":")[2])
    bundle = await _action_load(callback, userbot_manager, peer_id)
    if bundle is None:
        return
    _client, _owner, contact, messages, provider, heavy = bundle

    draft = await draft_reply(
        provider,
        contact,
        messages,
        heavy=heavy,
        global_style=_owner.global_style_profile,
        owner_id=_owner.id,
    )
    payload = json.dumps({"peer_id": peer_id, "text": draft}, ensure_ascii=False)

    from src.db.repo import create_pending_action

    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        action = await create_pending_action(
            session, user_id=owner.id, kind="send_message", payload=payload
        )

    kb = InlineKeyboardBuilder()
    kb.row(
        InlineKeyboardButton(
            text="✅ Отправить", callback_data=f"send:confirm:{action.id}"
        ),
        InlineKeyboardButton(
            text="❌ Отмена", callback_data=f"send:cancel:{action.id}"
        ),
    )
    if callback.message:
        await callback.message.edit_text(
            f"💬 <b>Черновик ответа — {contact.display_name}</b>\n\n{draft}\n\n"
            f"Отправить?",
            reply_markup=kb.as_markup(),
        )


@router.callback_query(F.data.startswith("chat:catchup:"))
async def cb_catchup(callback: CallbackQuery, userbot_manager: UserbotManager) -> None:
    peer_id = int(callback.data.split(":")[2])
    bundle = await _action_load(callback, userbot_manager, peer_id)
    if bundle is None:
        return
    _client, _owner, contact, messages, provider, heavy = bundle

    text = await catchup(
        provider,
        contact,
        messages,
        heavy=heavy,
        global_style=_owner.global_style_profile,
        owner_id=_owner.id,
    )
    if callback.message:
        await callback.message.edit_text(
            f"⏪ <b>Где мы остановились — {contact.display_name}</b>\n\n{text}",
            reply_markup=_actions_keyboard(peer_id),
        )


@router.message(Command("sync"))
async def cmd_sync(message: Message, userbot_manager: UserbotManager) -> None:
    """Sync метаданные диалогов + фоновый prefetch последних сообщений."""
    client = await _ensure_client(message, userbot_manager)
    if client is None:
        return
    from src.userbot.dialogs import prefetch_recent_messages, sync_dialogs

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
    stats = await sync_dialogs(client, owner, limit=500)
    total = sum(stats.values())
    await message.answer(
        f"✅ Синхронизировано {total} диалогов:\n"
        f"  👤 Люди: {stats['users']}\n"
        f"  🤖 Боты: {stats['bots']}\n"
        f"  👥 Группы: {stats['chats']}\n"
        f"  📰 Каналы: {stats['channels']}\n"
        f"  🗂 Архивных: {stats['archived']} (по умолчанию исключаются)\n\n"
        f"⏳ Фоном: подгружаю последние сообщения из топ-30 активных чатов "
        f"для мгновенного локального поиска. Это разово, дальше всё пишется в реальном времени."
    )

    async def _bg_prefetch() -> None:
        try:
            ps = await prefetch_recent_messages(
                client,
                message.from_user.id,
                top_n=30,
                per_chat=50,
                skip_channels=False,
            )
            await message.answer(
                f"📥 Prefetch готов: {ps['chats']} чатов, {ps['messages']} сообщений в БД."
            )
            # после prefetch — предложить извлечь память из топ-чатов
            auto_mem = getattr(owner.settings, "auto_extract_memories", False)
            if auto_mem:
                # авто-режим: дёргаем без вопроса
                await _auto_extract_memories(message, client, owner)
            else:
                await _offer_memory_extraction(message)

        except Exception:
            logger.exception("prefetch failed")
            await message.answer("⚠ Prefetch завершился с ошибкой — см. логи.")

    asyncio.create_task(_bg_prefetch())


async def _offer_memory_extraction(message: Message) -> None:
    """Предлагает извлечь память из топ-чатов после prefetch."""
    from src.db.repo import list_contacts

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        contacts = await list_contacts(
            session,
            owner,
            kinds=("user",),
            include_archived=False,
        )

    people = [c for c in contacts[:15] if not c.is_bot]
    if not people:
        return

    kb = InlineKeyboardBuilder()
    for c in people[:8]:
        kb.row(
            InlineKeyboardButton(
                text=f"🧠 {c.display_name}",
                callback_data=f"sync:mem:{c.peer_id}:{message.from_user.id}",
            )
        )
    kb.row(
        InlineKeyboardButton(
            text="🧠 Все контакты", callback_data=f"sync:mem:all:{message.from_user.id}"
        ),
        InlineKeyboardButton(
            text="❌ Пропустить", callback_data=f"sync:mem:skip:{message.from_user.id}"
        ),
    )
    await message.answer(
        "🧠 <b>Извлечь факты в память?</b>\n\n"
        "Выбери контакты, из чатов с которыми извлечь важные факты (отношения, договорённости, эмоции). "
        "Или «Все контакты» — обработаю всех людей.",
        reply_markup=kb.as_markup(),
    )


@router.callback_query(F.data.startswith("sync:mem:"))
async def cb_extract_memories(
    callback: CallbackQuery, userbot_manager: UserbotManager
) -> None:
    _, _, target, caller_id = callback.data.split(":")
    if int(caller_id) != callback.from_user.id:
        await callback.answer("Не твоя кнопка", show_alert=True)
        return

    if target == "skip":
        if callback.message:
            await callback.message.edit_text("Ок, пропустил.")
        await callback.answer()
        return

    if callback.message:
        await callback.message.edit_text("🧠 Извлекаю факты из переписок…")

    client = userbot_manager.get_client(callback.from_user.id)
    if client is None:
        await callback.answer("Сначала /login", show_alert=True)
        return

    from src.db.repo import list_contacts
    from src.llm.router import build_provider

    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        provider = await build_provider(session, owner)
        if provider is None:
            if callback.message:
                await callback.message.edit_text("Не задан LLM-ключ.")
            return

        if target == "all":
            contacts = await list_contacts(
                session, owner, kinds=("user",), include_archived=False
            )
            targets = [c for c in contacts[:20] if not c.is_bot]
        else:
            contact = await get_contact(session, owner, int(target))
            targets = [contact] if contact else []

    if not targets:
        if callback.message:
            await callback.message.edit_text("Нет подходящих контактов.")
        await callback.answer()
        return

    total = 0
    for ct in targets:
        try:
            messages = await load_chat(
                client, callback.from_user.id, ct.peer_id, limit=80
            )
            count = await extract_and_save_memories(provider, owner.id, ct, messages)
            if count:
                total += count
        except Exception:
            logger.exception("memory extraction failed for %s", ct.display_name)

    if callback.message:
        await callback.message.edit_text(
            f"✅ Извлечено <b>{total}</b> фактов из {len(targets)} контактов."
        )
    await callback.answer()


async def _auto_extract_memories(message: Message, client, owner) -> None:
    """Авто-извлечение памяти без вопроса (fire-and-forget)."""
    from src.db.repo import list_contacts
    from src.llm.router import build_provider
    import asyncio

    async with get_session() as session:
        contacts = await list_contacts(
            session, owner, kinds=("user",), include_archived=False
        )
        provider = await build_provider(session, owner)
    if provider is None:
        return

    targets = [c for c in contacts[:10] if not c.is_bot]
    if not targets:
        return

    total = 0
    for ct in targets:
        try:
            msgs = await load_chat(client, message.from_user.id, ct.peer_id, limit=60)
            count = await extract_and_save_memories(provider, owner.id, ct, msgs)
            total += count
        except Exception:
            pass

    if total:
        await message.answer(
            f"🧠 Авто-память: +{total} фактов из {len(targets)} контактов."
        )


@router.message(Command("recent"))
async def cmd_recent(message: Message) -> None:
    """Показать сводку по последней активности в чатах."""
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        from src.db.repo import list_contacts, fetch_chat_messages

        contacts = await list_contacts(
            session, owner, kinds=("user",), include_archived=False
        )
        active = [c for c in contacts if not c.is_bot]

    if not active:
        await message.answer("Нет активных чатов.")
        return

    lines = []
    for ct in active[:10]:
        async with get_session() as session:
            msgs = await fetch_chat_messages(session, owner, ct.peer_id, limit=3)
        last_date = msgs[0].date.strftime("%d.%m %H:%M") if msgs else "?"
        last_msg = (
            msgs[0].text or msgs[0].transcript or f"[{msgs[0].kind}]" if msgs else "?"
        )
        if len(last_msg) > 50:
            last_msg = last_msg[:47] + "…"
        who = "→" if msgs and msgs[0].is_outgoing else "←"
        lines.append(f"<b>{ct.display_name}</b> {who} {last_date}\n<i>{last_msg}</i>")

    body = "\n\n".join(lines)
    await message.answer(f"📋 <b>Последняя активность</b>\n\n{body}")


@router.callback_query(F.data.startswith("chat:story:"))
async def cb_story(callback: CallbackQuery) -> None:
    """Показать историю отношений с контактом."""
    peer_id = int(callback.data.split(":")[2])
    from src.core.memory_chain import build_chain_narrative

    narrative = await build_chain_narrative(peer_id, callback.from_user.id)
    if callback.message:
        if narrative:
            await callback.message.edit_text(
                narrative,
                reply_markup=_actions_keyboard(peer_id),
            )
        else:
            await callback.message.edit_text(
                "Недостаточно данных для истории (нужно минимум 3 факта).",
                reply_markup=_actions_keyboard(peer_id),
            )
    await callback.answer()


@router.callback_query(F.data.startswith("chat:analyze:"))
async def cb_analyze(callback: CallbackQuery) -> None:
    """Показать инструкцию по /analyze."""
    await callback.answer()
    await callback.message.answer(
        "🧠 <b>Полный анализ</b>\n\n"
        "Используй команду /analyze для полного анализа всех чатов.\n"
        "Или <code>/analyze Работа Семья</code> — только для указанных папок."
    )
