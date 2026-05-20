"""Обработчики памяти: store, forget, list, extract, check + inline callback'и."""

import logging

from aiogram import F, Router
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder

from src.core.infra.text_sanitizer import sanitize_html
from src.db.repo import (
    add_memory,
    add_memory_candidate,
    delete_memory,
    get_contact,
    get_or_create_user,
    list_memories,
    search_memories,
)
from src.db.session import get_session
from src.userbot import get_active_telethon_client

from .free_text_common import (
    _fire_record_trajectory,
    _summarize_intent_for_memory,
    memory_quick_keyboard,
)

logger = logging.getLogger(__name__)
router = Router(name="free_text_memory")


# ── exec-функции (вызываются из _dispatch в free_text.py) ──────────────


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
        client = get_active_telethon_client(message.from_user.id)
        if client is not None:
            from src.core.contacts.contact_resolver import resolve

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
            await message.answer(sanitize_html(f"🧠 Запомнил: <i>{fact}</i>"))
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
                sanitize_html(
                    f"📬 Сохранил как черновик: <i>{fact}</i>\n"
                    f"Подтверди через <code>/memory --inbox</code>"
                )
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
        client = get_active_telethon_client(message.from_user.id)
        if client is not None:
            from src.core.contacts.contact_resolver import resolve

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
            # переоткрываем owner в текущей сессии (detach-safe)
            owner2 = await get_or_create_user(session, message.from_user.id)
            await delete_memory(session, owner2, m.id)

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
        client = get_active_telethon_client(message.from_user.id)
        if client is not None:
            from src.core.contacts.contact_resolver import resolve

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

    from src.core.contacts.contact_resolver import resolve

    candidates = await resolve(client, owner, contact_name)
    if not candidates:
        await message.answer("Не нашёл такого контакта.")
        return

    peer_id = candidates[0].peer_id

    from src.core.contacts.chat_service import load_chat, message_to_text
    from src.core.memory.memory_queue import enqueue, MemoryJob

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
        await message.answer(
            sanitize_html(f"🤔 {question}"), reply_markup=kb.as_markup()
        )


# ── Memory callbacks ───────────────────────────────────────────────────


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
        client = get_active_telethon_client(callback.from_user.id)
        if client is not None:
            async with get_session() as session:
                owner = await get_or_create_user(session, callback.from_user.id)
            from src.core.contacts.contact_resolver import resolve

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
