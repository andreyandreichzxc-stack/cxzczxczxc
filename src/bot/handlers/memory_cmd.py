import logging

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from src.bot.filters import OwnerOnly
from src.core.contact_resolver import resolve
from src.core.memory_fuel import (
    format_depleted_contacts,
    format_fuel_line,
    get_fuel_stats,
)
from src.db.repo import (
    add_memory,
    delete_memory,
    get_linked_memories,
    get_memory_stats,
    get_or_create_user,
    list_memories,
    search_memories,
)
from src.db.session import get_session
from src.userbot.manager import UserbotManager


logger = logging.getLogger(__name__)
router = Router(name="memory_cmd")
router.message.filter(OwnerOnly())


@router.message(Command("memory"))
async def cmd_memory(message: Message, userbot_manager: UserbotManager) -> None:
    """Показать память — всё или про конкретный контакт."""
    args = (message.text or "").replace("/memory", "").strip()

    story_mode = "--story" in args
    if story_mode:
        args = args.replace("--story", "").strip()

    contact_id = None
    label = ""
    if args:
        contact_name = args.strip()
        async with get_session() as session:
            owner = await get_or_create_user(session, message.from_user.id)
        client = (
            userbot_manager.get_client(message.from_user.id)
            if userbot_manager
            else None
        )
        if client is not None:
            candidates = await resolve(client, owner, contact_name)
            if candidates:
                contact_id = candidates[0].peer_id
                label = f" — {candidates[0].label()}"

    if story_mode:
        if contact_id:
            from src.core.memory_chain import build_chain_narrative

            narrative = await build_chain_narrative(contact_id, message.from_user.id)
            if narrative:
                await message.answer(narrative)
            else:
                await message.answer(
                    "Недостаточно данных для истории (нужно минимум 3 факта)."
                )
        else:
            await message.answer("Укажи контакт: <code>/memory --story имя</code>")
        return

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        items = await list_memories(session, owner, contact_id=contact_id)
        stats = await get_memory_stats(session, owner)

    if not items:
        await message.answer("Память пуста.")
        return

    # Статистика
    pos = stats["by_sentiment"].get("positive", 0)
    neg = stats["by_sentiment"].get("negative", 0)
    neu = stats["by_sentiment"].get("neutral", 0)
    stat_line = f"🧠 <b>Память{label}</b>: {stats['total']} фактов ({pos} позитивных, {neg} негативных, {neu} нейтральных)\n"

    # Индикатор топлива памяти
    fuel = await get_fuel_stats(message.from_user.id)
    fuel_line = format_fuel_line(fuel)
    fuel_depleted = format_depleted_contacts(fuel)

    # Группировка по sentiment
    positive_lines: list[str] = []
    negative_lines: list[str] = []
    neutral_lines: list[str] = []

    for m in items:
        date_str = m.created_at.strftime("%d.%m.%Y") if m.created_at else "?"
        sent = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(
            m.sentiment or "", "⚪"
        )
        rel_icon = {
            "cause": "🎯",
            "effect": "⚡",
            "contradicts": "⚠️",
            "supports": "✅",
            "continues": "➡️",
            "example_of": "📌",
        }.get(m.relation_type or "", "")
        rel_prefix = f"{rel_icon} " if rel_icon else ""
        line = f"• {sent} [{date_str}]{rel_prefix} {m.fact}"
        if m.sentiment == "positive":
            positive_lines.append(line)
        elif m.sentiment == "negative":
            negative_lines.append(line)
        else:
            neutral_lines.append(line)

    body_parts = [stat_line, fuel_line]
    if fuel_depleted:
        body_parts.append(fuel_depleted)
    if positive_lines:
        body_parts.append(f"\n<b>🟢 Позитивные ({len(positive_lines)}):</b>")
        body_parts.extend(positive_lines[:10])
    if negative_lines:
        body_parts.append(f"\n<b>🔴 Негативные ({len(negative_lines)}):</b>")
        body_parts.extend(negative_lines[:10])
    if neutral_lines:
        body_parts.append(f"\n<b>⚪ Нейтральные ({len(neutral_lines)}):</b>")
        body_parts.extend(neutral_lines[:10])

    body = "\n".join(body_parts)

    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="🗑 Очистить негативные",
                    callback_data="memory:clear_negative",
                ),
                InlineKeyboardButton(
                    text="📊 Статистика", callback_data="memory:stats"
                ),
            ]
        ]
    )
    await message.answer(body, reply_markup=kb)


@router.callback_query(F.data == "memory:clear_negative")
async def cb_memory_clear_negative(callback: CallbackQuery) -> None:
    """Удалить все негативные факты."""
    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        items = await list_memories(session, owner)
        removed = 0
        for m in items:
            if m.sentiment == "negative":
                await delete_memory(session, owner, m.id)
                removed += 1
    if callback.message:
        await callback.message.edit_text(f"🧹 Удалено {removed} негативных фактов.")
    await callback.answer(f"Удалено {removed}")


@router.callback_query(F.data == "memory:stats")
async def cb_memory_stats(callback: CallbackQuery) -> None:
    """Показать детальную статистику памяти."""
    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        stats = await get_memory_stats(session, owner)

    lines = [
        "📊 <b>Статистика памяти</b>",
        "",
        f"🧠 Всего фактов: {stats['total']}",
        "",
        "<b>По тональности:</b>",
    ]
    for sentiment, count in stats["by_sentiment"].items():
        emoji = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(
            sentiment, "⚪"
        )
        lines.append(f"  {emoji} {sentiment}: {count}")
    lines.extend(
        [
            "",
            f"<b>По источникам:</b>",
        ]
    )
    for source, count in stats["by_source"].items():
        lines.append(f"  📄 {source}: {count}")
    lines.extend(
        [
            "",
            f"🎯 Высокая уверенность (≥0.8): {stats['high_confidence']}",
            f"👤 Связано с контактами: {stats['with_contact']}",
        ]
    )

    # Индикатор топлива памяти
    fuel = await get_fuel_stats(callback.from_user.id)
    lines.append("")
    lines.append(format_fuel_line(fuel))
    depleted_text = format_depleted_contacts(fuel)
    if depleted_text:
        lines.append(depleted_text)

    if callback.message:
        await callback.message.answer("\n".join(lines))
    await callback.answer()


@router.message(Command("remember"))
async def cmd_remember(
    message: Message, command: CommandObject, userbot_manager: UserbotManager
) -> None:
    """Вручную сохранить факт. /remember Настя злится из-за дедлайна"""
    args = (command.args or "").strip()
    if not args:
        await message.answer(
            "Использование: <code>/remember [контакт] факт</code>\nПример: <code>/remember Настя злится</code>"
        )
        return

    # пробуем отделить имя контакта от факта
    contact_name = None
    fact = args
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
    client = (
        userbot_manager.get_client(message.from_user.id) if userbot_manager else None
    )
    if client is not None:
        candidates = await resolve(client, owner, args)
        if candidates and candidates[0].score >= 70:
            contact_name = candidates[0].label()
            # пытаемся отделить: берём первое слово как имя
            words = args.split(None, 1)
            if len(words) > 1:
                fact = words[1]

    contact_id = None
    if contact_name:
        candidates = await resolve(client, owner, contact_name)
        if candidates:
            contact_id = candidates[0].peer_id

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        mem = await add_memory(
            session, owner, fact=fact, contact_id=contact_id, source="user"
        )

    await message.answer(f"🧠 Запомнил: <i>{fact}</i>")


@router.message(Command("insights"))
async def cmd_insights(message: Message) -> None:
    from src.core.memory_patterns import detect_patterns, format_insights

    insights = await detect_patterns(message.from_user.id)
    text, keyboards = format_insights(insights)
    # Если инсайтов нет — шлём один текст
    if not insights:
        await message.answer(text)
        return
    # Если есть — каждый инсайт отдельным сообщением с клавиатурой
    for ins, kb in zip(insights[:5], keyboards):
        detail = f"<b>{ins['title']}</b>\n{ins['detail']}\n💡 {ins['action']}"
        await message.answer(detail, reply_markup=kb)


@router.message(Command("forget"))
async def cmd_forget(
    message: Message, command: CommandObject, userbot_manager: UserbotManager
) -> None:
    """Удалить факты по подстроке. /forget злится"""
    args = (command.args or "").strip()
    if not args:
        await message.answer("Использование: <code>/forget часть текста</code>")
        return

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        found = await search_memories(session, owner, args)

    if not found:
        await message.answer("Ничего не нашёл.")
        return

    async with get_session() as session:
        for m in found:
            await delete_memory(session, owner, m.id)

    names = ", ".join(
        f"«{m.fact[:50]}…»" if len(m.fact) > 50 else f"«{m.fact}»" for m in found
    )
    await message.answer(f"🗑 Забыл: {names}")


@router.message(Command("archetypes"))
async def cmd_archetypes(message: Message) -> None:
    """Показать архетипы всех контактов."""
    from src.core.contact_archetypes import (
        classify_all_contacts,
        format_archetype_stats,
    )

    await message.answer("🏷 Анализирую контакты...")
    stats = await classify_all_contacts(message.from_user.id)
    text = format_archetype_stats(stats)
    await message.answer(text)


@router.callback_query(F.data.startswith("pattern:"))
async def cb_pattern_action(callback: CallbackQuery) -> None:
    """Обрабатывает нажатия на inline-кнопки паттернов."""
    data = callback.data.split(":")
    action = data[1]  # remind, dismiss, history, write
    contact_id = int(data[2]) if len(data) > 2 else 0

    if action == "dismiss":
        if callback.message:
            await callback.message.edit_text(
                callback.message.text + "\n\n🔕 Ок, не сейчас."
            )
        await callback.answer()
        return

    if action == "remind":
        from src.db.repo import get_contact, get_or_create_user

        async with get_session() as session:
            owner = await get_or_create_user(session, callback.from_user.id)
            contact = await get_contact(session, owner, contact_id)
            name = contact.display_name if contact else str(contact_id)
            # Сохраняем факт в память
            await add_memory(
                session,
                owner,
                fact=f"Пользователь хочет напоминание о созвоне с {name}",
                source="user",
                sentiment="neutral",
            )
        if callback.message:
            await callback.message.edit_text(
                f"📅 Напоминание для <b>{name}</b>\n"
                f"Напиши: <code>/remind за час до созвона с {name}</code>"
            )
        await callback.answer(f"Напоминание для {name}")
        return

    if action == "history":
        await callback.answer(
            f"История контакта {contact_id} — открой /chat {contact_id} или /memory"
        )
        return

    if action == "write":
        await callback.answer("Напиши: /send контакт текст")
        return

    await callback.answer()
