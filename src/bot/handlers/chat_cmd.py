import asyncio
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
from src.bot.handlers.rate_limiter import check_rate_limit
from src.core.infra.text_sanitizer import sanitize_html
from src.core.services.chat_actions import (
    catchup_action,
    draft_reply_action,
    extract_tasks_action,
    get_chat_message_count,
    summarize_chat_action,
)
from src.core.contacts.chat_service import load_chat
from src.core.memory.memory_extractor import extract_and_save_memories
from src.core.contacts.contact_resolver import ContactCandidate, resolve
from src.db.repo import (
    add_watched_peer,
    get_contact,
    get_or_create_user,
    get_watched_peers,
    is_peer_watched,
    list_memories,
    remove_watched_peer,
)
from src.db.session import get_session
from src.llm.router import build_provider
from src.userbot.manager import UserbotManager


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
            text="📝 Саммари", callback_data=f"chat:summary:{peer_id}:50"
        ),
        InlineKeyboardButton(
            text="✅ Задачи/обещания", callback_data=f"chat:tasks:{peer_id}:50"
        ),
    )
    kb.row(
        InlineKeyboardButton(
            text="💬 Черновик ответа", callback_data=f"chat:draft:{peer_id}:50"
        ),
        InlineKeyboardButton(
            text="⏪ Где мы остановились", callback_data=f"chat:catchup:{peer_id}:50"
        ),
    )
    kb.row(
        InlineKeyboardButton(text="📖 История", callback_data=f"chat:story:{peer_id}"),
        InlineKeyboardButton(
            text="👤 Профиль", callback_data=f"chat:profile:{peer_id}"
        ),
    )
    kb.row(
        InlineKeyboardButton(text="👁 Следить", callback_data=f"chat:watch:{peer_id}"),
        InlineKeyboardButton(
            text="👁 Не следить", callback_data=f"chat:unwatch:{peer_id}"
        ),
    )
    kb.row(
        InlineKeyboardButton(text="🧠 Полный анализ", callback_data="chat:analyze:all"),
    )
    kb.row(
        InlineKeyboardButton(
            text="🔢 Выбрать лимит", callback_data=f"chat:limit:{peer_id}"
        ),
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
    # ── Rate-limit ────────────────────────────────────────────────────
    if not await check_rate_limit(message.from_user.id, window=5, max_requests=10):
        await message.answer("Слишком часто. Подожди.")
        return

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
    parts = callback.data.split(":")
    if len(parts) < 3:
        await callback.answer("Ошибка данных.", show_alert=True)
        return
    peer_id = int(parts[2])
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


@router.callback_query(F.data.startswith("chat:watch:"))
async def cb_watch(callback: CallbackQuery) -> None:
    parts = callback.data.split(":")
    if len(parts) < 3:
        await callback.answer("Ошибка данных.", show_alert=True)
        return
    peer_id = int(parts[2])
    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        contact = await get_contact(session, owner, peer_id)
        if await is_peer_watched(session, owner, peer_id):
            await callback.answer("Уже слежу за этим чатом 👁", show_alert=True)
            return
        await add_watched_peer(session, owner, peer_id)
        name = contact.display_name if contact else str(peer_id)

    await callback.answer(f"Теперь слежу за чатом «{name}» 👁", show_alert=True)


@router.callback_query(F.data.startswith("chat:unwatch:"))
async def cb_unwatch(callback: CallbackQuery) -> None:
    parts = callback.data.split(":")
    if len(parts) < 3:
        await callback.answer("Ошибка данных.", show_alert=True)
        return
    peer_id = int(parts[2])
    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        contact = await get_contact(session, owner, peer_id)
        if not await is_peer_watched(session, owner, peer_id):
            await callback.answer("Я и так не слежу за этим чатом", show_alert=True)
            return
        await remove_watched_peer(session, owner, peer_id)
        name = contact.display_name if contact else str(peer_id)

    await callback.answer(f"Больше не слежу за чатом «{name}»", show_alert=True)


@router.callback_query(F.data.startswith("chat:summary:"))
async def cb_summary(callback: CallbackQuery, userbot_manager: UserbotManager) -> None:
    parts = callback.data.split(":")
    peer_id = int(parts[2])
    limit = int(parts[3]) if len(parts) >= 4 else 50
    await callback.answer("Подгружаю чат…")
    if callback.message:
        await callback.message.edit_text("⏳ Подгружаю последние сообщения…")

    result = await summarize_chat_action(
        callback.from_user.id, peer_id, userbot_manager, limit=limit
    )
    if result is None or callback.message is None:
        return
    await callback.message.edit_text(result.html, reply_markup=result.markup)


@router.callback_query(F.data.startswith("chat:tasks:"))
async def cb_tasks(callback: CallbackQuery, userbot_manager: UserbotManager) -> None:
    parts = callback.data.split(":")
    peer_id = int(parts[2])
    limit = int(parts[3]) if len(parts) >= 4 else 50
    await callback.answer("Извлекаю задачи…")
    if callback.message:
        await callback.message.edit_text("⏳ Анализирую переписку…")

    result = await extract_tasks_action(
        callback.from_user.id, peer_id, userbot_manager, limit=limit
    )
    if result is None or callback.message is None:
        return
    await callback.message.edit_text(result.html, reply_markup=result.markup)


@router.callback_query(F.data.startswith("chat:draft:"))
async def cb_draft(callback: CallbackQuery, userbot_manager: UserbotManager) -> None:
    parts = callback.data.split(":")
    peer_id = int(parts[2])
    limit = int(parts[3]) if len(parts) >= 4 else 50
    await callback.answer("Готовлю черновик…")
    if callback.message:
        await callback.message.edit_text("⏳ Пишу черновик ответа…")

    result = await draft_reply_action(
        callback.from_user.id, peer_id, userbot_manager, limit=limit
    )
    if result is None or callback.message is None:
        return
    await callback.message.edit_text(result.html, reply_markup=result.markup)


@router.callback_query(F.data.startswith("chat:catchup:"))
async def cb_catchup(callback: CallbackQuery, userbot_manager: UserbotManager) -> None:
    parts = callback.data.split(":")
    peer_id = int(parts[2])
    limit = int(parts[3]) if len(parts) >= 4 else 50
    await callback.answer("Подгружаю историю…")
    if callback.message:
        await callback.message.edit_text("⏳ Ищу где вы остановились…")

    result = await catchup_action(
        callback.from_user.id, peer_id, userbot_manager, limit=limit
    )
    if result is None or callback.message is None:
        return
    await callback.message.edit_text(result.html, reply_markup=result.markup)


@router.callback_query(F.data.startswith("chat:limit:"))
async def cb_limit(callback: CallbackQuery) -> None:
    """Показывает меню выбора лимита сообщений."""
    parts = callback.data.split(":")
    if len(parts) < 3:
        await callback.answer("Ошибка данных.", show_alert=True)
        return
    peer_id = int(parts[2])

    # Проверяем количество сообщений
    try:
        count = await get_chat_message_count(callback.from_user.id, peer_id)
    except Exception:
        count = 0

    # Строим клавиатуру выбора лимита
    kb = InlineKeyboardBuilder()

    if count > 0:
        status = (
            f"В чате <b>{count}</b> сообщений."
            if count <= 100
            else f"⚠️ Чат большой: <b>{count}</b> сообщений."
        )
        if count > 100:
            status += "\nАнализ может занять время."
    else:
        status = "Сообщений пока нет или чат не синхронизирован."

    # Кнопки лимита
    options = [50, 100, 200, 500]
    row = []
    for n in options:
        if n <= count or count == 0:  # показываем все опции если нет данных
            row.append(
                InlineKeyboardButton(
                    text=str(n), callback_data=f"chat:summary:{peer_id}:{n}"
                )
            )
    kb.row(*row)

    kb.row(
        InlineKeyboardButton(
            text="📋 Все сообщения",
            callback_data=f"chat:summary:{peer_id}:{max(count, 500)}",
        )
    )
    kb.row(InlineKeyboardButton(text="🔙 Назад", callback_data=f"chat:pick:{peer_id}"))

    if callback.message:
        await callback.message.edit_text(
            f"{status}\n\nСколько сообщений проанализировать?",
            reply_markup=kb.as_markup(),
        )
    await callback.answer()


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
        f"  🗂 Архивных: {stats['archived']}"
        f"{' (удалено ' + str(stats.get('removed', 0)) + ' неактуальных) ' if stats.get('removed') else ''}\n\n"
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
    parts = callback.data.split(":")
    if len(parts) < 4:
        await callback.answer("Ошибка данных.", show_alert=True)
        return
    _, _, target, caller_id = parts[:4]
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
            count = await extract_and_save_memories(
                provider, callback.from_user.id, ct, messages
            )
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
    """Авто-извлечение памяти без вопроса (fire-and-forget).

    При холодном старте (warmup) извлекает из ВСЕХ контактов.
    В штатном режиме — только из top-N (memory_warmup_max_contacts).
    """
    from src.db.repo import list_contacts
    from src.core.memory.memory_warmup import should_full_extract
    from src.config import settings

    async with get_session() as session:
        contacts = await list_contacts(
            session, owner, kinds=("user",), include_archived=False
        )
        provider = await build_provider(session, owner)
    if provider is None:
        return

    # --- Warmup: все контакты при холодном старте ---
    telegram_id = message.from_user.id
    in_warmup = should_full_extract(
        telegram_id,
        idle_timeout_sec=settings.memory_warmup_idle_timeout_sec,
    )

    max_contacts = len(contacts) if in_warmup else settings.memory_warmup_max_contacts
    targets = [c for c in contacts[:max_contacts] if not c.is_bot]
    if not targets:
        return

    logger.info(
        "auto_extract: %s mode, %d contacts",
        "warmup" if in_warmup else "normal",
        len(targets),
    )

    total = 0
    for ct in targets:
        try:
            msgs = await load_chat(client, message.from_user.id, ct.peer_id, limit=60)
            count = await extract_and_save_memories(provider, telegram_id, ct, msgs)
            total += count
        except Exception:
            logger.exception("auto extract memories failed")

    if total:
        mode_label = "🔥 Warmup" if in_warmup else "🧠"
        await message.answer(
            f"{mode_label} Авто-память: +{total} фактов из {len(targets)} контактов."
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
    await message.answer(sanitize_html(f"📋 <b>Последняя активность</b>\n\n{body}"))


@router.message(Command("watchlist"))
async def cmd_watchlist(message: Message) -> None:
    """Показать список отслеживаемых чатов."""
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        watched = await get_watched_peers(session, owner)

    if not watched:
        await message.answer(
            "👁 <b>Отслеживание чатов</b>\n\n"
            "Пока список пуст — я сохраняю <b>все</b> чаты.\n\n"
            "Добавь чаты через /chat → «Следить», "
            "и я буду сохранять только их.\n\n"
            "<i>Используй /watchlist снова для просмотра списка.</i>"
        )
        return

    # Получаем имена контактов
    lines = [f"👁 <b>Отслеживаемые чаты ({len(watched)})</b>\n"]
    kb = InlineKeyboardBuilder()

    async with get_session() as session:
        for pid in sorted(watched):
            contact = await get_contact(session, owner, pid)
            name = contact.display_name if contact else f"ID:{pid}"
            icon = {"user": "👤", "chat": "👥", "channel": "📢", "bot": "🤖"}.get(
                contact.peer_kind if contact else "user", "💬"
            )
            lines.append(f"{icon} <b>{name}</b>")
            kb.row(
                InlineKeyboardButton(
                    text=f"❌ Перестать следить за «{name}»",
                    callback_data=f"chat:unwatch:{pid}",
                )
            )

    kb.row(InlineKeyboardButton(text="➕ Добавить чат", callback_data="watchlist:add"))

    await message.answer(
        "\n".join(lines),
        reply_markup=kb.as_markup(),
    )


@router.callback_query(F.data == "watchlist:add")
async def cb_watchlist_add(callback: CallbackQuery) -> None:
    await callback.answer()
    if callback.message:
        await callback.message.edit_text(
            "Чтобы добавить чат в отслеживание:\n\n"
            "1. Используй <code>/chat Имя</code> для поиска контакта\n"
            "2. Нажми <b>👁 Следить</b> в меню действий\n\n"
            "Или введи <code>/chat</code> для списка недавних контактов."
        )


@router.callback_query(F.data.startswith("chat:story:"))
async def cb_story(callback: CallbackQuery) -> None:
    """Показать историю отношений с контактом."""
    parts = callback.data.split(":")
    if len(parts) < 3:
        await callback.answer("Ошибка данных.", show_alert=True)
        return
    peer_id = int(parts[2])
    from src.core.memory.memory_chain import build_chain_narrative

    narrative = await build_chain_narrative(peer_id, callback.from_user.id)
    if callback.message:
        if narrative:
            await callback.message.edit_text(  # type: ignore[union-attr]
                narrative,
                reply_markup=_actions_keyboard(peer_id),
            )
        else:
            await callback.message.edit_text(  # type: ignore[union-attr]
                "Недостаточно данных для истории (нужно минимум 3 факта).",
                reply_markup=_actions_keyboard(peer_id),
            )
    await callback.answer()


@router.callback_query(F.data.startswith("chat:profile:"))
async def cb_profile(callback: CallbackQuery, userbot_manager: UserbotManager) -> None:
    """Показать профиль контакта."""
    parts = callback.data.split(":")
    if len(parts) < 3:
        await callback.answer("Ошибка данных.", show_alert=True)
        return
    peer_id = int(parts[2])
    client = userbot_manager.get_client(callback.from_user.id)
    if client is None:
        await callback.answer("Сначала /login", show_alert=True)
        return

    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        contact = await get_contact(session, owner, peer_id)

    from src.core.contacts.contact_resolver import ContactCandidate

    _ = ContactCandidate(
        peer_id=peer_id,
        display_name=contact.display_name if contact else str(peer_id),
        username=None,
        peer_kind="user",
        score=100,
    )

    contact_name = contact.display_name if contact else str(peer_id)
    if callback.message:
        await callback.message.answer(
            f"👤 Используй /profile {contact_name} для просмотра профиля"
        )
    else:
        await callback.answer("Сообщение недоступно.", show_alert=True)


@router.callback_query(F.data.startswith("chat:analyze:"))
async def cb_analyze(callback: CallbackQuery) -> None:
    """Показать инструкцию по /analyze."""
    await callback.answer()
    if callback.message:
        await callback.message.answer(
            "🧠 <b>Полный анализ</b>\n\n"
            "Используй команду /analyze для полного анализа всех чатов.\n"
            "Или <code>/analyze Работа Семья</code> — только для указанных папок."
        )
    else:
        await callback.answer("Сообщение недоступно.", show_alert=True)
