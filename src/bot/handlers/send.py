import json
import logging
import re

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder

from src.bot.filters import OwnerOnly
from src.bot.handlers.rate_limiter import check_rate_limit
from src.bot.handlers.smart_keyboard import smart_post_action_keyboard
from src.core.contacts.contact_resolver import ContactCandidate, resolve
from src.core.infra.text_sanitizer import sanitize_html
from src.db.repo import (
    create_pending_action,
    delete_pending_action,
    get_contact,
    get_or_create_user,
    get_pending_action,
)
from src.db.session import get_session
from src.llm.base import ChatMessage
from src.llm.router import build_provider
from src.userbot import get_active_telethon_client
from src.userbot.manager import UserbotManager


logger = logging.getLogger(__name__)
router = Router(name="send")
router.message.filter(OwnerOnly())
router.callback_query.filter(OwnerOnly())


class SendStates(StatesGroup):
    waiting_edit = State()


PARSE_SYSTEM = (
    "Тебе дают свободную фразу-инструкцию вида «скажи Оле, что созвон в 8».\n"
    "Извлеки получателя и текст сообщения. Сообщение должно быть готово к отправке "
    "(в первом лице, без префиксов «передай», «скажи»).\n\n"
    'Возвращай ТОЛЬКО JSON: {"recipient": "...", "message": "..."}.\n'
    "Если не удаётся определить — верни поля null."
)


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\s*|\s*```$", "", text).strip()
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        return {}


def _confirm_keyboard(action_id: int):
    kb = InlineKeyboardBuilder()
    kb.row(
        InlineKeyboardButton(
            text="✅ Отправить", callback_data=f"send:confirm:{action_id}"
        ),
        InlineKeyboardButton(text="✏ Изменить", callback_data=f"send:edit:{action_id}"),
    )
    kb.row(
        InlineKeyboardButton(text="❌ Отмена", callback_data=f"send:cancel:{action_id}")
    )
    return kb.as_markup()


def _candidates_keyboard(candidates: list[ContactCandidate], message_text: str):
    """Кнопки выбора получателя для send. callback_data: send:pick:<peer_id>:<action_id>
    Action создаётся уже после выбора, поэтому здесь храним сообщение в коротком кэше через FSM-data."""
    kb = InlineKeyboardBuilder()
    for c in candidates:
        kb.row(
            InlineKeyboardButton(
                text=f"{c.label()} · {c.score}",
                callback_data=f"send:pick:{c.peer_id}",
            )
        )
    kb.row(InlineKeyboardButton(text="❌ Отмена", callback_data="send:cancel:0"))
    return kb.as_markup()


@router.message(Command("send"))
async def cmd_send(
    message: Message,
    command: CommandObject,
    state: FSMContext,
    userbot_manager: UserbotManager,
) -> None:
    # ── Rate-limit ────────────────────────────────────────────────────
    if not await check_rate_limit(message.from_user.id, window=5, max_requests=10):
        await message.answer("Слишком часто. Подожди.")
        return

    client = userbot_manager.get_client(message.from_user.id)
    if client is None:
        await message.answer("Сначала /login.")
        return
    raw = (command.args or "").strip()
    if not raw:
        await message.answer(
            "Использование: <code>/send скажи Оле, что созвон в 8</code>\n"
            "Или: <code>/send @username | текст сообщения</code>"
        )
        return

    recipient_query: str | None = None
    text: str | None = None

    if "|" in raw:
        parts = raw.split("|", 1)
        recipient_query = parts[0].strip()
        text = parts[1].strip()

    if not recipient_query or not text:
        async with get_session() as session:
            owner = await get_or_create_user(session, message.from_user.id)
            provider = await build_provider(session, owner)
        if provider is None:
            await message.answer(
                "Нужен LLM-ключ для NL-парсинга. Добавь в /settings или используй формат «получатель | текст»."
            )
            return
        parsed_raw = await provider.chat(
            [
                ChatMessage(role="system", content=PARSE_SYSTEM),
                ChatMessage(role="user", content=raw),
            ],
            heavy=False,
        )
        parsed = _parse_json(parsed_raw)
        recipient_query = parsed.get("recipient") or recipient_query
        text = parsed.get("message") or text

    if not recipient_query or not text:
        await message.answer(
            "Не удалось разобрать запрос. Попробуй формат: <code>/send Оля | текст</code>."
        )
        return

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
    candidates = await resolve(client, owner, recipient_query)
    if not candidates:
        await message.answer(
            sanitize_html(
                f"Не нашёл контакт «{recipient_query}». Запусти /sync и попробуй снова."
            )
        )
        return

    if len(candidates) == 1 or candidates[0].score >= 90:
        await _create_and_confirm(
            message,
            owner_telegram_id=message.from_user.id,
            peer_id=candidates[0].peer_id,
            text=text,
            label=candidates[0].label(),
        )
        return

    await state.set_data({"send_text": text})
    await message.answer(
        sanitize_html(f"Кому именно отправить «<i>{text[:80]}</i>»?"),
        reply_markup=_candidates_keyboard(candidates, text),
    )


async def _create_and_confirm(
    message: Message,
    *,
    owner_telegram_id: int,
    peer_id: int,
    text: str,
    label: str,
) -> None:
    payload = json.dumps({"peer_id": peer_id, "text": text}, ensure_ascii=False)
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_telegram_id)
        action = await create_pending_action(
            session, user_id=owner.id, kind="send_message", payload=payload
        )
    guard_hint = ""
    try:
        from src.core.contacts.send_guard import build_send_guard

        guard = await build_send_guard(owner_telegram_id, peer_id, text)
        if guard.formatted_html:
            guard_hint = "\n\n" + guard.formatted_html
    except Exception:
        logger.warning("send guard failed", exc_info=True)

    await message.answer(
        sanitize_html(
            f"🤔 <b>Готов отправить</b>\n\n→ <b>Кому:</b> {label}\n→ <b>Текст:</b>\n{text}{guard_hint}\n\n<i>Подтверди отправку 👇</i>"
        ),
        reply_markup=_confirm_keyboard(action.id),
    )


@router.callback_query(F.data.startswith("send:pick:"))
async def cb_pick(callback: CallbackQuery, state: FSMContext) -> None:
    parts = callback.data.split(":")
    if len(parts) < 3:
        await callback.answer("Ошибка данных.", show_alert=True)
        return
    peer_id = int(parts[2])
    data = await state.get_data()
    text = data.get("send_text")
    if not text:
        await callback.answer("Сессия потеряна, попробуй /send заново", show_alert=True)
        return
    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        contact = await get_contact(session, owner, peer_id)
    label = contact.display_name if contact else str(peer_id)
    guard_hint = ""
    try:
        from src.core.contacts.send_guard import build_send_guard

        guard = await build_send_guard(callback.from_user.id, peer_id, text)
        if guard.formatted_html:
            guard_hint = "\n\n" + guard.formatted_html
    except Exception:
        logger.warning("send guard failed", exc_info=True)

    payload = json.dumps({"peer_id": peer_id, "text": text}, ensure_ascii=False)
    async with get_session() as session:
        owner = await get_or_create_user(session, callback.from_user.id)
        action = await create_pending_action(
            session, user_id=owner.id, kind="send_message", payload=payload
        )

    await state.clear()
    if callback.message:
        await callback.message.edit_text(
            sanitize_html(
                f"🤔 <b>Готов отправить</b>\n\n"
                f"→ <b>Кому:</b> {label}\n"
                f"→ <b>Текст:</b>\n{text}{guard_hint}"
            ),
            reply_markup=_confirm_keyboard(action.id),
        )
    await callback.answer()


@router.callback_query(F.data.startswith("send:cancel:"))
async def cb_cancel(callback: CallbackQuery, state: FSMContext) -> None:
    parts = callback.data.split(":")
    action_id = int(parts[2]) if len(parts) > 2 else 0
    if action_id:
        async with get_session() as session:
            user = await get_or_create_user(session, callback.from_user.id)
            await delete_pending_action(session, action_id, user)
    await state.clear()
    if callback.message:
        await callback.message.edit_text("❌ Отправка отменена. 🚫")
    await callback.answer()


@router.callback_query(F.data.startswith("send:edit:"))
async def cb_edit(callback: CallbackQuery, state: FSMContext) -> None:
    parts = callback.data.split(":")
    if len(parts) < 3:
        await callback.answer("Ошибка данных.", show_alert=True)
        return
    action_id = int(parts[2])
    await state.set_state(SendStates.waiting_edit)
    await state.set_data({"action_id": action_id})
    await callback.message.answer("Введи новый текст сообщения. /cancel — отмена.")
    await callback.answer()


@router.message(SendStates.waiting_edit)
async def step_edit(message: Message, state: FSMContext) -> None:
    new_text = (message.text or "").strip()
    if not new_text:
        await message.answer("Пустой текст. Повтори или /cancel.")
        return
    data = await state.get_data()
    action_id = data.get("action_id")
    async with get_session() as session:
        user = await get_or_create_user(session, message.from_user.id)
        action = await get_pending_action(session, action_id, user)
        if action is None:
            await state.clear()
            await message.answer("Сессия отправки потеряна. Запусти /send заново.")
            return
        payload = json.loads(action.payload)
        payload["text"] = new_text
        action.payload = json.dumps(payload, ensure_ascii=False)
        peer_id = payload["peer_id"]

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        contact = await get_contact(session, owner, peer_id)
    label = contact.display_name if contact else str(peer_id)
    guard_hint = ""
    try:
        from src.core.contacts.send_guard import build_send_guard

        guard = await build_send_guard(message.from_user.id, peer_id, new_text)
        if guard.formatted_html:
            guard_hint = "\n\n" + guard.formatted_html
    except Exception:
        logger.warning("send guard failed", exc_info=True)

    await state.clear()
    await message.answer(
        sanitize_html(
            f"🤔 <b>Готов отправить</b>\n\n→ <b>Кому:</b> {label}\n→ <b>Текст:</b>\n{new_text}{guard_hint}"
        ),
        reply_markup=_confirm_keyboard(action_id),
    )


@router.callback_query(F.data.startswith("send:confirm:"))
async def cb_confirm(callback: CallbackQuery, userbot_manager: UserbotManager) -> None:
    parts = callback.data.split(":")
    if len(parts) < 3:
        await callback.answer("Ошибка данных.", show_alert=True)
        return
    action_id = int(parts[2])
    client = userbot_manager.get_client(callback.from_user.id)
    if client is None:
        await callback.answer("Сначала /login", show_alert=True)
        return

    async with get_session() as session:
        user = await get_or_create_user(session, callback.from_user.id)
        action = await get_pending_action(session, action_id, user)
        if action is None:
            await callback.answer(
                "Действие не найдено или уже выполнено", show_alert=True
            )
            return
        payload = json.loads(action.payload)
        peer_id = payload["peer_id"]
        text = payload["text"]
        await delete_pending_action(session, action_id, user)

    # Send Guard — предупреждение перед отправкой
    from src.core.contacts.send_guard import build_send_guard, store_undo

    guard = await build_send_guard(callback.from_user.id, peer_id, text)
    if guard.warnings:
        await callback.answer(f"⚠️ {guard.warnings[0][:100]}", show_alert=True)

    sent_msg = None
    try:
        entity = await client.get_entity(peer_id)
        sent_msg = await client.send_message(entity, text)
    except Exception as e:
        logger.exception("send_message failed")
        await callback.answer("Ошибка при отправке", show_alert=True)
        if callback.message:
            await callback.message.edit_text(
                sanitize_html(f"❌ Не удалось отправить 😞: <code>{e}</code>")
            )
        return

    # Сохраняем для undo
    if sent_msg:
        await store_undo(callback.from_user.id, peer_id, sent_msg.id, text)

    # Получаем имя контакта для красивого отчёта
    label = str(peer_id)
    async with get_session() as session:
        user = await get_or_create_user(session, callback.from_user.id)
        contact = await get_contact(session, user, peer_id)
        if contact:
            label = contact.display_name

    snippet = (text or "")[:60]
    if len(text or "") > 60:
        snippet += "…"

    after_kb = smart_post_action_keyboard("send", {"peer_id": str(peer_id)})

    if callback.message:
        await callback.message.edit_text(
            sanitize_html(f"✅ Отправлено «{label}»: {snippet}"), reply_markup=after_kb
        )
    await callback.answer("Отправлено")


@router.callback_query(F.data.startswith("send:undo:"))
async def cb_undo(
    callback: CallbackQuery, userbot_manager: UserbotManager | None = None
) -> None:
    from src.core.contacts.send_guard import get_undo

    parts = callback.data.split(":")
    peer_id = int(parts[2])
    message_id = int(parts[3]) if len(parts) > 3 else None

    client = (
        userbot_manager.get_client(callback.from_user.id) if userbot_manager else None
    )
    if not client:
        client = get_active_telethon_client(callback.from_user.id)
    if not client:
        await callback.answer("Сначала /login", show_alert=True)
        return

    if message_id is None:
        # fallback: use get_undo (old callback format without message_id)
        undo = await get_undo(callback.from_user.id)
        if undo:
            peer_id, message_id = undo[0], undo[1]
        else:
            await callback.answer("Слишком поздно для отмены (60с)", show_alert=True)
            return

    try:
        entity = await client.get_entity(peer_id)
        await client.delete_messages(entity=entity, message_ids=[message_id])
        if callback.message:
            await callback.message.edit_text("↩ Сообщение отменено.")
        await callback.answer("Отменено")
    except Exception as e:
        await callback.answer(f"Не удалось отменить: {e}", show_alert=True)


@router.callback_query(F.data.startswith("send:again:"))
async def cb_send_again(callback: CallbackQuery) -> None:
    await callback.answer("Открой меню отправки")
    if callback.message:
        await callback.message.edit_text(
            "✏️ Используй /chat или отправь сообщение напрямую чтобы написать ещё.",
            reply_markup=smart_post_action_keyboard("general"),
        )
