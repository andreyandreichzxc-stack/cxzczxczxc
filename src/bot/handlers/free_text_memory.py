"""Обработчики памяти: store, forget, list, extract, check + inline callback'и."""

import logging

from aiogram import F, Router
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder

from src.bot.filters import OwnerOnly
from src.core.infra.text_sanitizer import sanitize_html
from src.db.repo import (
    add_memory,
    add_memory_candidate,
    delete_memory,
    get_contact,
    get_graph_stats,
    get_or_create_user,
    link_memories,
    list_memories,
    search_memories,
)
from src.db.session import get_session
from src.userbot import get_active_telethon_client

from .free_text_common import safe_answer


logger = logging.getLogger(__name__)
router = Router(name="free_text_memory")
router.message.filter(OwnerOnly())
router.callback_query.filter(OwnerOnly())


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
            await add_memory(
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
    await message.answer(sanitize_html(f"🗑 Забыл: {names}"))


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
    await message.answer(sanitize_html(f"🧠 <b>Память{label}</b>\n\n{body}"))


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


# ── Новые intent-хендлеры (Phase 5.2) ────────────────────────────────


async def _exec_update_memory(intent, message) -> None:
    """Update an existing memory fact by ID or search query."""
    from src.db.models import Memory

    memory_id = intent.get("memory_id")
    query = (intent.get("query") or "").strip()
    new_fact = (intent.get("new_fact") or "").strip()
    new_sentiment = (intent.get("new_sentiment") or "").strip()
    new_importance = intent.get("new_importance")

    if not memory_id and not query:
        await message.answer("🤷 Укажи ID факта или поисковый запрос для обновления.")
        return
    if not new_fact:
        await message.answer("🤷 Что написать вместо старого факта? Укажи new_fact.")
        return

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)

        target = None
        if memory_id:
            target = await session.get(Memory, int(memory_id))
            if target is None or target.user_id != owner.id or not target.is_active:
                target = None
        elif query:
            found = await search_memories(session, owner, query)
            if found:
                target = found[0]

        if target is None:
            await message.answer("🙅 Не нашёл такой факт в памяти.")
            return

        # Обновляем поля
        target.fact = new_fact
        if new_sentiment in ("positive", "negative", "neutral"):
            target.sentiment = new_sentiment
        if new_importance is not None:
            try:
                target.importance = float(new_importance)
            except (TypeError, ValueError):
                pass
        await session.flush()

    await safe_answer(
        message,
        sanitize_html(f"✏️ Факт #{target.id} обновлён:\n<i>{new_fact}</i>"),
    )


async def _exec_link_memories(intent, message) -> None:
    """Create a relationship between two memory facts."""
    source_id = intent.get("source_id")
    target_id = intent.get("target_id")
    relation_type = (intent.get("relation_type") or "").strip() or None

    if not source_id or not target_id:
        await message.answer("🤷 Укажи source_id и target_id — ID двух фактов.")
        return

    try:
        source_id = int(source_id)
        target_id = int(target_id)
    except (TypeError, ValueError):
        await message.answer("❌ source_id и target_id должны быть числами.")
        return

    if source_id == target_id:
        await message.answer("🤷 Нельзя связать факт с самим собой.")
        return

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        link = await link_memories(
            session,
            owner,
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
        )

    if link is None:
        await message.answer("🙅 Один из фактов не найден или не принадлежит тебе.")
        return

    rel = f" ({relation_type})" if relation_type else ""
    await safe_answer(
        message,
        sanitize_html(f"🔗 Факты #{source_id} ↔ #{target_id} связаны{rel}."),
    )


async def _exec_show_memory_health(intent, message) -> None:
    """Display memory health metrics."""
    from src.core.memory.memory_health import calculate_health_score

    health = await calculate_health_score(message.from_user.id)

    emoji = health.get("emoji", "🟡")
    label = health.get("label", "Средне")
    score = health["score"]

    lines = [f"{emoji} <b>Здоровье памяти:</b> {score}/100 — {label}", ""]
    # Компоненты
    lines.append(f"  📊 Уверенность:      {health.get('confidence_score', 0)}/100")
    lines.append(f"  📊 Покрытие:         {health.get('coverage_score', 0)}/100")
    lines.append(f"  📊 Свежесть:         {health.get('freshness_score', 0)}/100")
    lines.append(f"  📊 Структура:        {health.get('structure_score', 0)}/100")
    lines.append(f"  📊 Теги:             {health.get('tag_score', 0)}/100")
    lines.append("")
    lines.append(f"  📦 Всего фактов:      {health.get('total_facts', 0)}")
    lines.append(f"  👤 Контактов:        {health.get('total_contacts', 0)}")
    # Диагностика
    diag = health.get("diagnostics", [])
    if diag:
        lines.append("")
        lines.append("💡 <b>Рекомендации:</b>")
        for d in diag[:3]:
            lines.append(f"  • {d}")

    await safe_answer(message, sanitize_html("\n".join(lines)))


async def _exec_show_memory_graph(intent, message) -> None:
    """Display memory graph statistics."""
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        stats = await get_graph_stats(session, owner.id)

    lines = [
        "🔗 <b>Граф памяти</b>",
        "",
        f"  Узлов:                {stats['node_count']}",
        f"  Рёбер (всего):        {stats['total_edges']}",
        f"  Средняя степень:      {stats['avg_degree']:.2f}",
        f"  Компонент связности:  {stats['components']}",
        f"  Изолированных узлов:  {stats['isolated_nodes']}",
    ]

    # Типы связей
    ebt = stats.get("edges_by_type", {})
    if ebt:
        lines.append("")
        lines.append("📌 <b>Типы связей:</b>")
        for rel_type, cnt in sorted(ebt.items(), key=lambda x: -x[1]):
            lines.append(f"  {rel_type}: {cnt}")

    # Top-хабы
    hubs = stats.get("top_hubs", [])
    if hubs:
        lines.append("")
        lines.append("⭐ <b>Ключевые узлы:</b>")
        for h in hubs[:3]:
            fact = sanitize_html(h.get("fact", "")[:80])
            lines.append(f"  #{h['memory_id']} (degree {h['degree']}): {fact}")

    await safe_answer(message, "\n".join(lines))


async def _exec_show_sessions(intent, message) -> None:
    """Show recent conversation sessions."""
    from src.core.memory.session_recorder import get_session_history

    limit = intent.get("limit", 5)

    async with get_session() as session:
        history = await get_session_history(
            session, message.from_user.id, limit=int(limit)
        )

    if not history:
        await safe_answer(message, "📭 Нет записанных сессий.")
        return

    lines = ["📋 <b>Последние сессии:</b>", ""]
    for s in history:
        start = s.get("started_at", "?")[:16] if s.get("started_at") else "?"
        sid = s.get("session_id", "?")
        turns = s.get("turn_count", 0)
        summary = s.get("summary", "")
        line = f"  #{sid} — {start} ({turns} сообщ.)"
        if summary:
            line += f"\n    📝 {sanitize_html(summary[:100])}"
        lines.append(line)

    await safe_answer(message, "\n".join(lines))


async def _exec_show_suggestions(intent, message) -> None:
    """Show proactive memory-based patterns and suggestions."""
    from src.core.memory.memory_patterns import detect_patterns

    patterns = await detect_patterns(message.from_user.id)

    if not patterns:
        await safe_answer(message, "💡 Нет активных паттернов или предложений.")
        return

    lines = ["💡 <b>Паттерны и предложения:</b>", ""]
    for p in patterns:
        title = sanitize_html(p.get("title", p.get("type", "?")))
        detail = sanitize_html(p.get("detail", ""))
        action = sanitize_html(p.get("action", ""))
        lines.append(f"• <b>{title}</b>")
        if detail:
            lines.append(f"  {detail}")
        if action:
            lines.append(f"  🔹 {action}")
        lines.append("")

    await safe_answer(message, "\n".join(lines))


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
            lines.append(f"{emoji} {sanitize_html(m.fact[:100])}")
        lines.append(f"\n<i>Всего: {len(memories)} фактов. /memory — подробнее</i>")
        await callback.message.answer(sanitize_html("\n".join(lines)))
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
