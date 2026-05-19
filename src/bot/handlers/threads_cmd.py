"""Команда /threads — просмотр активных переписок (inbox)."""

import logging

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from src.bot.filters import OwnerOnly
from src.db.repo import (
    fetch_chat_messages,
    get_contact,
    get_or_create_user,
    list_active_conversations,
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
async def cmd_threads(message: Message) -> None:
    """Показать активные переписки."""
    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        convos = await list_active_conversations(session, owner, limit=20)
        if not convos:
            await message.answer("📭 Нет активных переписок.")
            return

        lines = ["<b>📬 Активные переписки</b>", ""]
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
        text += "\n\n<i>Нажми на кнопку для действий</i>"
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
    lines.append("")
    lines.append("<i>Ответь в Telegram или через /send</i>")
    await callback.message.answer("\n".join(lines))


@router.callback_query(F.data.startswith("thread:reply:"))
async def cb_thread_reply(callback: CallbackQuery, userbot_manager=None) -> None:
    """Сгенерировать черновик ответа для треда."""
    peer_id = int(callback.data.split(":")[2])
    from src.core.summarizer import draft_reply
    from src.core.text_sanitizer import sanitize_html
    from src.llm.router import build_provider

    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        contact = await get_contact(session, owner, peer_id)
        provider = await build_provider(session, owner)
        msgs = await fetch_chat_messages(session, owner, peer_id, limit=20)
        name = contact.display_name if contact else str(peer_id)

        if provider and contact and msgs:
            draft = await draft_reply(
                provider, contact, msgs, heavy=False, global_style=None
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
