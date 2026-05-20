"""Команда /threads — просмотр активных переписок (inbox)."""

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
from src.core.memory.memory_fuel import format_fuel_line, get_fuel_stats
from src.core.memory.memory_neighbors import format_neighbors, get_neighbors
from src.db.repo import (
    fetch_chat_messages,
    get_contact,
    get_linked_memories,
    get_or_create_user,
    list_active_conversations,
    list_contacts,
    list_folders,
    list_memories,
)
from src.db.session import get_session

logger = logging.getLogger(__name__)

router = Router(name="threads_cmd")
router.message.filter(OwnerOnly())
router.callback_query.filter(OwnerOnly())

STATUS_EMOJI = {
    "active": "🟢",
    "waiting_reply": "🟡",
    "snoozed": "💤",
    "closed": "⚫",
}


@router.message(Command("threads"))
async def cmd_threads(message: Message, command: CommandObject | None = None) -> None:
    """Показать активные переписки. /threads — все, /threads Работа — по папке."""
    # Индикатор топлива памяти
    fuel = await get_fuel_stats(message.from_user.id)
    fuel_line = format_fuel_line(fuel)

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)

        folder_name: str | None = None
        matched: str | None = None
        if command and command.args:
            folder_name = command.args.strip()

        if folder_name:
            folders = await list_folders(session, owner)
            folder_titles = {f.title.lower(): f.title for f in folders}
            matched = folder_titles.get(folder_name.lower())
            if not matched:
                available = (
                    ", ".join(f.title for f in folders) if folders else "нет папок"
                )
                await message.answer(
                    f"❌ Папка «{folder_name}» не найдена.\nДоступные: {available}"
                )
                return

            # Фильтруем контакты по папке
            contacts = await list_contacts(
                session, owner, kinds=("user",), include_bots=False
            )
            peer_ids = {
                c.peer_id
                for c in contacts
                if c.folder_names and matched in c.folder_names.split(",")
            }
            convos = await list_active_conversations(session, owner, limit=50)
            convos = [c for c in convos if c.peer_id in peer_ids][:20]

            title = f"📬 Папка «{matched}» — активные переписки"
        else:
            convos = await list_active_conversations(session, owner, limit=20)
            title = "<b>📬 Активные переписки</b>"

        if not convos:
            if folder_name:
                await message.answer(f"📭 В папке «{matched}» нет активных переписок.")
            else:
                await message.answer("📭 Нет активных переписок.")
            return

        lines = [title, "", fuel_line, ""]
        kb_rows = []
        for i, conv in enumerate(convos[:15]):
            contact = await get_contact(session, owner, conv.peer_id)
            name = contact.display_name if contact else str(conv.peer_id)
            emoji = STATUS_EMOJI.get(conv.status, "⚪")
            unread = f"({conv.unread_count})" if conv.unread_count else ""
            lines.append(f"{emoji} <b>{name}</b> {unread}")
            kb_rows.append(
                [
                    InlineKeyboardButton(
                        text=f"💬 {name[:15]}",
                        callback_data=f"thread:open:{conv.peer_id}",
                    ),
                    InlineKeyboardButton(
                        text="✍️ Ответить",
                        callback_data=f"thread:reply:{conv.peer_id}",
                    ),
                ]
            )
        text = "\n".join(lines)
        text += "\n\n<i>👆 Нажми на кнопку для действий</i>"
        kb = InlineKeyboardMarkup(
            inline_keyboard=kb_rows
            + [
                [
                    InlineKeyboardButton(
                        text="🔄 Обновить", callback_data="thread:refresh"
                    )
                ]
            ]
        )
        await message.answer(text, reply_markup=kb)


@router.callback_query(F.data == "thread:refresh")
async def cb_thread_refresh(callback: CallbackQuery) -> None:
    """Обновить список активных переписок."""
    await cmd_threads(callback.message)  # type: ignore[arg-type]
    await callback.answer("Обновлено")


@router.callback_query(F.data.startswith("thread:open:"))
async def cb_thread_open(callback: CallbackQuery) -> None:
    """Показать последние сообщения из треда."""
    peer_id = int(callback.data.split(":")[2])
    await callback.answer(f"Открыть чат {peer_id} — открой в Telegram")

    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        contact = await get_contact(session, owner, peer_id)
        name = contact.display_name if contact else str(peer_id)
        msgs = await fetch_chat_messages(session, owner, peer_id, limit=5)

    lines = [f"<b>💬 {name}</b> — последние сообщения:", ""]
    for m in msgs:
        direction = "→" if m.is_outgoing else "←"
        sender = "Вы" if m.is_outgoing else (m.sender_name or name)
        txt = (m.text or m.transcript or "")[:100]
        lines.append(f"{direction} <b>{sender}:</b> {txt}")

    # Показываем факты памяти о контакте
    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        memories = await list_memories(session, owner, contact_id=peer_id)
        if memories:
            lines.append("")
            lines.append("<b>🧠 Память о контакте:</b>")
            for m in memories[:5]:
                rel_icon = {
                    "cause": "🎯",
                    "effect": "⚡",
                    "contradicts": "⚠️",
                    "supports": "✅",
                    "continues": "➡️",
                    "example_of": "📌",
                }.get(m.relation_type or "", "")
                prefix = f"{rel_icon} " if rel_icon else "• "
                lines.append(f"{prefix}{m.fact}")

            # Загружаем связанные факты (если есть хоть один с relation_type)
            linked = await get_linked_memories(session, owner, memories[0].id, limit=3)
            if linked:
                lines.append("")
                lines.append("<b>🔗 Связанные факты:</b>")
                for lm in linked[:3]:
                    rel_label = {
                        "cause": "причина",
                        "effect": "следствие",
                        "contradicts": "противоречие",
                        "supports": "подтверждение",
                        "continues": "продолжение",
                        "example_of": "пример",
                    }.get(lm.relation_type or "", lm.relation_type or "связь")
                    lines.append(f"  {lm.relation_type}: «{lm.fact}»")

            # Семантические соседи для первого факта
            neighbors = await get_neighbors(
                callback.from_user.id, memories[0].id, limit=2
            )
            n_text = format_neighbors(neighbors)
            if n_text:
                lines.append("")
                lines.append(n_text)

            # Компактная история отношений
            from src.core.memory.memory_chain import build_chain_narrative

            narrative = await build_chain_narrative(peer_id, callback.from_user.id)
            if narrative:
                lines.append("")
                n_src = narrative.split("\n")
                lines.extend(n_src[: min(len(n_src), 10)])

    lines.append("")
    lines.append("<i>Ответь в Telegram или через /send</i>")
    await callback.message.answer("\n".join(lines))


@router.callback_query(F.data.startswith("thread:reply:"))
async def cb_thread_reply(callback: CallbackQuery, userbot_manager=None) -> None:
    """Сгенерировать черновик ответа для треда."""
    peer_id = int(callback.data.split(":")[2])
    from src.core.intelligence.summarizer import draft_reply
    from src.core.infra.text_sanitizer import sanitize_html
    from src.llm.router import build_provider

    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        contact = await get_contact(session, owner, peer_id)
        provider = await build_provider(session, owner)
        msgs = await fetch_chat_messages(session, owner, peer_id, limit=20)
        name = contact.display_name if contact else str(peer_id)

        if provider and contact and msgs:
            draft = await draft_reply(
                provider,
                contact,
                msgs,
                heavy=False,
                global_style=None,
                owner_id=owner.id,
            )
            if draft:
                safe_draft = sanitize_html(draft)[:350]
                kb = InlineKeyboardMarkup(
                    inline_keyboard=[
                        [
                            InlineKeyboardButton(
                                text="📝 Открыть /chat",
                                callback_data=f"thread:chat:{peer_id}",
                            )
                        ]
                    ]
                )
                await callback.message.answer(
                    f"✍️ <b>Черновик для {name}:</b>\n\n{safe_draft}\n\n"
                    f"<i>Отправь через /send {name} твой текст</i>",
                    reply_markup=kb,
                )
                return

    await callback.answer("Не удалось сгенерировать черновик")
