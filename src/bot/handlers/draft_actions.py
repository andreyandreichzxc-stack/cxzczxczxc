"""Callback handlers for draft suggestion inline keyboard (send/edit/ignore/variants)."""

from __future__ import annotations

import hashlib
import logging
import time

from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from src.agents.draft_agent import draft_variants
from src.bot.filters import OwnerOnly
from src.bot.handlers.smart_keyboard import smart_post_action_keyboard
from src.bot.states import DraftStates
from src.core.contacts.send_guard import store_undo
from src.db.repo import get_or_create_user as _get_or_create_user
from src.db.session import get_session
from src.llm.router import build_provider
from src.userbot import get_active_telethon_client


logger = logging.getLogger(__name__)

router = Router(name="draft_actions")
router.message.filter(OwnerOnly())
router.callback_query.filter(OwnerOnly())


# in-memory store: draft_hash -> (timestamp, full draft text)
_draft_texts: dict[str, tuple[float, str]] = {}

# variant_groups: hash -> (timestamp, peer_id, contact_name, incoming_text, variant_dicts)
_variant_groups: dict[str, tuple[float, int, str, str, list[dict]]] = {}
DRAFT_TTL_SECONDS = 30 * 60  # 30 минут


def _draft_cleanup() -> None:
    """Удаляет черновики старше DRAFT_TTL_SECONDS."""
    now = time.time()
    stale = [k for k, (ts, _) in _draft_texts.items() if now - ts > DRAFT_TTL_SECONDS]
    for k in stale:
        del _draft_texts[k]
    # Clean variant groups too
    stale_v = [k for k, v in _variant_groups.items() if now - v[0] > DRAFT_TTL_SECONDS]
    for k in stale_v:
        del _variant_groups[k]


def store_draft(draft_text: str) -> str:
    """Сохраняет черновик и возвращает hash-ключ для callback'ов."""
    draft_hash = hashlib.sha256(draft_text.encode()).hexdigest()[:8]
    _draft_texts[draft_hash] = (time.time(), draft_text)
    _draft_cleanup()
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
    if len(parts) < 4:
        await callback.answer("Ошибка данных.", show_alert=True)
        return
    peer_id = int(parts[2])
    draft_hash = parts[3]

    draft_data = _draft_texts.pop(draft_hash, None)
    if draft_data is None:
        await callback.answer("Черновик устарел или не найден", show_alert=True)
        return
    ts, draft_text = draft_data
    if time.time() - ts > DRAFT_TTL_SECONDS:
        await callback.answer("Черновик устарел", show_alert=True)
        return

    client = get_active_telethon_client(callback.from_user.id)
    if client is None:
        await callback.answer("Нет активной сессии. Сначала /login.", show_alert=True)
        return

    try:
        entity = await client.get_entity(peer_id)
        sent_msg = await client.send_message(entity=entity, message=draft_text)
        if callback.message:
            await store_undo(callback.from_user.id, peer_id, sent_msg.id, draft_text)
            after_kb = smart_post_action_keyboard("send", {"peer_id": str(peer_id)})
            await callback.message.edit_text("✅ Отправлено! 🚀", reply_markup=after_kb)
    except ValueError as e:
        if callback.message:
            await callback.message.edit_text(f"❌ Ошибка 😞: {e}")
        else:
            await callback.answer(f"❌ Ошибка: {e}", show_alert=True)
    except Exception as e:
        from telethon.errors import FloodWaitError

        if isinstance(e, FloodWaitError):
            if callback.message:
                await callback.message.edit_text(f"❌ Flood wait ⏳: {e.seconds}с")
            else:
                await callback.answer(f"❌ Flood wait: {e.seconds}с", show_alert=True)
        else:
            await callback.answer(f"❌ Ошибка отправки: {e}", show_alert=True)
    await callback.answer()


# ── Игнорировать ──


@router.callback_query(F.data.startswith("draft:ignore:"))
async def cb_draft_ignore(callback: CallbackQuery) -> None:
    parts = callback.data.split(":")
    if len(parts) < 4:
        await callback.answer("Ошибка данных.", show_alert=True)
        return
    draft_hash = parts[3]
    _draft_texts.pop(draft_hash, None)
    if callback.message:
        await callback.message.edit_text("🗑 Пропущено")
    await callback.answer()


# ── Редактировать ──


@router.callback_query(F.data.startswith("draft:edit:"))
async def cb_draft_edit(callback: CallbackQuery, state: FSMContext) -> None:
    parts = callback.data.split(":")
    if len(parts) == 4:
        # Old format: draft:edit:{peer_id}:{hash} — single draft edit
        peer_id = int(parts[2])
        draft_hash = parts[3]
        await state.set_state(DraftStates.waiting_edit)
        await state.set_data({"peer_id": peer_id, "draft_hash": draft_hash})
        await callback.message.answer(
            "Пришли новый текст черновика для отправки. /cancel — отмена."
        )
    elif len(parts) == 3:
        # New format: draft:edit:{group_hash} — variant group edit
        group_hash = parts[2]
        group_data = _variant_groups.get(group_hash)
        if group_data is None:
            await callback.answer("Черновик устарел или не найден", show_alert=True)
            return
        ts, peer_id, contact_name, incoming_text, variants = group_data
        await state.set_state(DraftStates.waiting_edit)
        await state.set_data(
            {
                "peer_id": peer_id,
                "draft_hash": group_hash,
                "draft_variants": True,
            }
        )
        await callback.message.answer(
            f"Пришли новый текст для отправки контакту {contact_name}. /cancel — отмена."
        )
    else:
        await callback.answer("Неверный формат", show_alert=True)
        return
    await callback.answer()


@router.message(DraftStates.waiting_edit)
async def step_draft_edit(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    peer_id = data.get("peer_id")
    new_text = (message.text or "").strip()
    if not new_text:
        await message.answer("Пустой текст. Повтори или /cancel.")
        return

    client = get_active_telethon_client(message.from_user.id)
    if client is None:
        await message.answer("Нет активной сессии. Сначала /login.")
        await state.clear()
        return

    try:
        entity = await client.get_entity(peer_id)
        sent_msg = await client.send_message(entity=entity, message=new_text)
        await store_undo(message.from_user.id, peer_id, sent_msg.id, new_text)
        after_kb = smart_post_action_keyboard("edit", {"peer_id": str(peer_id)})
        await message.answer("✅ Отправлено! 🚀", reply_markup=after_kb)
    except Exception as e:
        await state.clear()
        await message.answer(f"❌ Ошибка отправки 😞: {e}")
    await state.clear()


# ── Variant group storage ────────────────────────────────────────────────


def store_variant_group(
    peer_id: int, contact_name: str, incoming_text: str, variants: list[dict]
) -> str:
    """Сохраняет группу вариантов и возвращает hash для callback'ов."""
    raw = f"{peer_id}:{contact_name}:{incoming_text}:{str(variants)}"
    group_hash = hashlib.sha256(raw.encode()).hexdigest()[:8]
    _variant_groups[group_hash] = (
        time.time(),
        peer_id,
        contact_name,
        incoming_text,
        variants,
    )
    _draft_cleanup()
    return group_hash


def build_variants_keyboard(
    group_hash: str, variants: list[dict]
) -> InlineKeyboardMarkup:
    """Строит inline-клавиатуру для выбора варианта."""
    buttons = [
        [
            InlineKeyboardButton(
                text=f"{i}️⃣ {v['tone']}", callback_data=f"draft:choose:{group_hash}:{i}"
            )
        ]
        for i, v in enumerate(variants, 1)
    ]
    buttons.append(
        [
            InlineKeyboardButton(
                text="🔄 Улучшить", callback_data=f"draft:improve:{group_hash}"
            ),
            InlineKeyboardButton(
                text="✏️ Править", callback_data=f"draft:edit:{group_hash}"
            ),
            InlineKeyboardButton(text="❌ Отмена", callback_data="draft:cancel"),
        ]
    )
    return InlineKeyboardMarkup(inline_keyboard=buttons)


async def show_draft_variants(
    callback: CallbackQuery, peer_id: int, contact_name: str, incoming_text: str
) -> None:
    """Генерирует 3 варианта черновика и показывает их с клавиатурой выбора."""
    async with get_session() as session:
        owner = await _get_or_create_user(session, callback.from_user.id)
        provider = await build_provider(session, owner)
    if provider is None:
        await callback.answer("Не задан LLM-ключ.", show_alert=True)
        return
    variants = await draft_variants(provider, contact_name, incoming_text)

    if not variants or len(variants) < 2:
        # Fallback to single draft
        from src.agents.draft_agent import draft

        single = await draft(provider, contact_name, incoming_text)
        variants = [{"tone": "черновик", "text": single["draft"]}]

    group_hash = store_variant_group(peer_id, contact_name, incoming_text, variants)

    lines = [f"🤖 <b>Черновики для {contact_name}:</b>\n"]
    for i, v in enumerate(variants, 1):
        lines.append(f"{i}️⃣ <b>{v['tone'].capitalize()}:</b> {v['text']}")

    html = "\n".join(lines)
    kb = build_variants_keyboard(group_hash, variants)

    await callback.message.edit_text(html, reply_markup=kb)
    await callback.answer()


# ── Choose variant ───────────────────────────────────────────────────────


@router.callback_query(F.data.startswith("draft:choose:"))
async def cb_draft_choose(callback: CallbackQuery) -> None:
    parts = callback.data.split(":")
    # draft:choose:{hash}:{idx}
    if len(parts) < 4:
        await callback.answer("Неверный формат данных", show_alert=True)
        return
    group_hash = parts[2]
    try:
        idx = int(parts[3]) - 1  # convert to 0-based
    except ValueError:
        await callback.answer("Неверный индекс", show_alert=True)
        return

    group_data = _variant_groups.get(group_hash)
    if group_data is None:
        await callback.answer("Черновик устарел или не найден", show_alert=True)
        return
    ts, peer_id, contact_name, incoming_text, variants = group_data
    if time.time() - ts > DRAFT_TTL_SECONDS:
        await callback.answer("Черновик устарел", show_alert=True)
        return
    if idx < 0 or idx >= len(variants):
        await callback.answer("Неверный вариант", show_alert=True)
        return

    draft_text = variants[idx]["text"]
    client = get_active_telethon_client(callback.from_user.id)
    if client is None:
        await callback.answer("Нет активной сессии. Сначала /login.", show_alert=True)
        return

    try:
        entity = await client.get_entity(peer_id)
        sent_msg = await client.send_message(entity=entity, message=draft_text)
        if callback.message:
            await store_undo(callback.from_user.id, peer_id, sent_msg.id, draft_text)
            after_kb = smart_post_action_keyboard("send", {"peer_id": str(peer_id)})
            await callback.message.edit_text("✅ Отправлено! 🚀", reply_markup=after_kb)
    except ValueError as e:
        if callback.message:
            await callback.message.edit_text(f"❌ Ошибка 😞: {e}")
        else:
            await callback.answer(f"❌ Ошибка: {e}", show_alert=True)
    except Exception as e:
        from telethon.errors import FloodWaitError

        if isinstance(e, FloodWaitError):
            if callback.message:
                await callback.message.edit_text(f"❌ Flood wait ⏳: {e.seconds}с")
            else:
                await callback.answer(f"❌ Flood wait: {e.seconds}с", show_alert=True)
        else:
            await callback.answer(f"❌ Ошибка отправки: {e}", show_alert=True)
    await callback.answer()


# ── Improve variant ──────────────────────────────────────────────────────


@router.callback_query(F.data.startswith("draft:improve:"))
async def cb_draft_improve(callback: CallbackQuery) -> None:
    parts = callback.data.split(":")
    if len(parts) < 3:
        await callback.answer("Неверный формат", show_alert=True)
        return
    group_hash = parts[2]

    group_data = _variant_groups.get(group_hash)
    if group_data is None:
        await callback.answer("Черновик устарел", show_alert=True)
        return
    ts, peer_id, contact_name, incoming_text, old_variants = group_data

    if callback.message:
        await callback.message.edit_text("🔄 Улучшаю черновики…")
    await callback.answer()

    async with get_session() as session:
        owner = await _get_or_create_user(session, callback.from_user.id)
        provider = await build_provider(session, owner)
    if provider is None:
        await callback.answer("Не задан LLM-ключ.", show_alert=True)
        return
    enriched_text = (
        f"{incoming_text}\n\n(сделай ответ живее, естественнее, как в разговорной речи)"
    )
    variants = await draft_variants(provider, contact_name, enriched_text)

    if not variants or len(variants) < 2:
        from src.agents.draft_agent import draft

        single = await draft(provider, contact_name, enriched_text)
        variants = [{"tone": "черновик", "text": single["draft"]}]

    _variant_groups.pop(group_hash, None)
    new_hash = store_variant_group(peer_id, contact_name, incoming_text, variants)

    lines = [f"🤖 <b>Улучшенные черновики для {contact_name}:</b>\n"]
    for i, v in enumerate(variants, 1):
        lines.append(f"{i}️⃣ <b>{v['tone'].capitalize()}:</b> {v['text']}")

    html = "\n".join(lines)
    kb = build_variants_keyboard(new_hash, variants)

    if callback.message:
        await callback.message.edit_text(html, reply_markup=kb)


# ── Cancel variants ──────────────────────────────────────────────────────


@router.callback_query(F.data == "draft:cancel")
async def cb_draft_cancel(callback: CallbackQuery) -> None:
    if callback.message:
        await callback.message.edit_text("❌ Отменено")
    await callback.answer()
