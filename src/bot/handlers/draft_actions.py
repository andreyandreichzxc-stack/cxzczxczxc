"""Callback handlers for draft suggestion inline keyboard (send/edit/ignore)."""

from __future__ import annotations

import hashlib
import logging

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from src.bot.filters import OwnerOnly
from src.bot.states import DraftStates
from src.userbot.manager import _MANAGER_SINGLETON


logger = logging.getLogger(__name__)

router = Router(name="draft_actions")
router.message.filter(OwnerOnly())
router.callback_query.filter(OwnerOnly())


# in-memory store: draft_hash -> full draft text
_draft_texts: dict[str, str] = {}


def store_draft(draft_text: str) -> str:
    """Сохраняет черновик и возвращает hash-ключ для callback'ов."""
    draft_hash = hashlib.sha256(draft_text.encode()).hexdigest()[:8]
    _draft_texts[draft_hash] = draft_text
    return draft_hash


def draft_keyboard(peer_id: int, draft_hash: str) -> InlineKeyboardMarkup:
    """Строит inline-клавиатуру для черновика."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="▶️ Отправить",
                    callback_data=f"draft:send:{peer_id}:{draft_hash}",
                ),
                InlineKeyboardButton(
                    text="✏️ Редактировать",
                    callback_data=f"draft:edit:{peer_id}:{draft_hash}",
                ),
            ],
            [
                InlineKeyboardButton(
                    text="🗑 Игнорировать",
                    callback_data=f"draft:ignore:{peer_id}:{draft_hash}",
                ),
            ],
        ]
    )


# ── Отправка ──


@router.callback_query(F.data.startswith("draft:send:"))
async def cb_draft_send(callback: CallbackQuery) -> None:
    parts = callback.data.split(":")
    peer_id = int(parts[2])
    draft_hash = parts[3]

    draft_text = _draft_texts.pop(draft_hash, None)
    if draft_text is None:
        await callback.answer("Черновик устарел или не найден", show_alert=True)
        return

    client = (
        _MANAGER_SINGLETON.get_client(callback.from_user.id)
        if _MANAGER_SINGLETON
        else None
    )
    if client is None:
        await callback.answer("Нет активной сессии. Сначала /login.", show_alert=True)
        return

    try:
        await client.send_message(entity=int(peer_id), message=draft_text)
        if callback.message:
            await callback.message.edit_text("✅ Отправлено")
    except ValueError as e:
        await callback.message.edit_text(f"❌ Ошибка: {e}")
    except Exception as e:
        from telethon.errors import FloodWaitError

        if isinstance(e, FloodWaitError):
            await callback.message.edit_text(f"❌ Flood wait: {e.seconds}с")
        else:
            await callback.answer(f"❌ Ошибка отправки: {e}", show_alert=True)
    await callback.answer()


# ── Игнорировать ──


@router.callback_query(F.data.startswith("draft:ignore:"))
async def cb_draft_ignore(callback: CallbackQuery) -> None:
    parts = callback.data.split(":")
    draft_hash = parts[3]
    _draft_texts.pop(draft_hash, None)
    if callback.message:
        await callback.message.edit_text("🗑 Пропущено")
    await callback.answer()


# ── Редактировать ──


@router.callback_query(F.data.startswith("draft:edit:"))
async def cb_draft_edit(callback: CallbackQuery, state: FSMContext) -> None:
    parts = callback.data.split(":")
    peer_id = int(parts[2])
    draft_hash = parts[3]

    await state.set_state(DraftStates.waiting_edit)
    await state.set_data({"peer_id": peer_id, "draft_hash": draft_hash})
    await callback.message.answer(
        "Пришли новый текст черновика для отправки. /cancel — отмена."
    )
    await callback.answer()


@router.message(DraftStates.waiting_edit)
async def step_draft_edit(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    peer_id = data.get("peer_id")
    new_text = (message.text or "").strip()
    if not new_text:
        await message.answer("Пустой текст. Повтори или /cancel.")
        return

    client = (
        _MANAGER_SINGLETON.get_client(message.from_user.id)
        if _MANAGER_SINGLETON
        else None
    )
    if client is None:
        await message.answer("Нет активной сессии. Сначала /login.")
        await state.clear()
        return

    try:
        await client.send_message(entity=int(peer_id), message=new_text)
        await message.answer("✅ Отправлено")
    except Exception as e:
        await message.answer(f"❌ Ошибка отправки: {e}")
    await state.clear()
