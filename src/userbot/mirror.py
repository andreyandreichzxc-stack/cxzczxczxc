"""Зеркало всех сообщений (входящих и исходящих) в БД и FTS5 в реальном времени.
Транскрипция голоса и парсинг документов — лениво в момент анализа."""

from __future__ import annotations

import logging
from datetime import datetime

from telethon import TelegramClient, events
from telethon.tl.custom import Message as TgMessage
from telethon.tl.types import User as TgUser

from src.core.notifier import notifier
from src.db.repo import get_or_create_user, upsert_message, upsert_contact
from src.db.session import get_session


logger = logging.getLogger(__name__)


def _classify(msg: TgMessage) -> str:
    if msg.voice:
        return "voice"
    if msg.audio:
        return "audio"
    if msg.document:
        return "document"
    if msg.photo:
        return "photo"
    if msg.video:
        return "video"
    if msg.sticker:
        return "sticker"
    if msg.video_note:
        return "video_note"
    if msg.poll:
        return "poll"
    if msg.geo:
        return "geo"
    if msg.venue:
        return "venue"
    if msg.contact:
        return "contact"
    if msg.game:
        return "game"
    if msg.invoice:
        return "invoice"
    if msg.text:
        return "text"
    return "other"


def _peer_id_of(msg: TgMessage) -> int | None:
    chat = msg.chat
    if chat is not None:
        return chat.id
    if msg.peer_id is not None and hasattr(msg.peer_id, "user_id"):
        return msg.peer_id.user_id
    return msg.chat_id


async def _sender_label(msg: TgMessage) -> str | None:
    if msg.out:
        return None  # это мы сами
    try:
        sender = await msg.get_sender()
    except Exception:
        sender = None
    if sender is None:
        return None
    parts = [getattr(sender, "first_name", None), getattr(sender, "last_name", None)]
    name = " ".join(p for p in parts if p).strip()
    if name:
        return name
    return getattr(sender, "username", None) or str(sender.id)


def attach_mirror(client: TelegramClient, owner_telegram_id: int) -> None:
    async def on_message(event: events.NewMessage.Event) -> None:
        try:
            msg: TgMessage = event.message
            peer_id = _peer_id_of(msg)
            if not peer_id:
                return

            kind = _classify(msg)
            text = msg.text or msg.message or None
            sender_name = await _sender_label(msg)

            # ===== SESSION 1: только быстрые DB-операции =====
            async with get_session() as session:
                owner = await get_or_create_user(session, owner_telegram_id)

                # сброс статуса отсутствия при любом исходящем
                if msg.out and owner.absence_status is not None:
                    owner.absence_status = None
                    owner.absence_message = None

                try:
                    chat = await event.get_chat()
                except Exception:
                    chat = None
                if chat is not None:
                    if isinstance(chat, TgUser):
                        parts = [
                            getattr(chat, "first_name", None),
                            getattr(chat, "last_name", None),
                        ]
                        display = " ".join(p for p in parts if p).strip() or (
                            chat.username or str(peer_id)
                        )
                        await upsert_contact(
                            session,
                            owner,
                            peer_id=peer_id,
                            peer_kind="user",
                            is_bot=bool(getattr(chat, "bot", False)),
                            display_name=display,
                            username=getattr(chat, "username", None),
                            phone=getattr(chat, "phone", None),
                        )
                    else:
                        title = getattr(chat, "title", None) or str(peer_id)
                        kind_chat = (
                            "channel" if getattr(chat, "broadcast", False) else "chat"
                        )
                        await upsert_contact(
                            session,
                            owner,
                            peer_id=peer_id,
                            peer_kind=kind_chat,
                            is_bot=False,
                            display_name=title,
                            username=getattr(chat, "username", None),
                        )

                await upsert_message(
                    session,
                    user_id=owner.id,
                    peer_id=peer_id,
                    message_id=msg.id,
                    sender_id=msg.sender_id,
                    sender_name=sender_name,
                    is_outgoing=bool(msg.out),
                    date=msg.date.replace(tzinfo=None)
                    if msg.date
                    else datetime.utcnow(),
                    kind=kind,
                    text=text,
                    transcript=None,
                    media_path=None,
                    extracted_text=None,
                )

                # детекция фраз отсутствия в исходящих сообщениях
                if msg.out and msg.text:
                    from src.core.absence_detector import detect_absence_phrases

                    status, message_text = detect_absence_phrases(msg.text)
                    if status:
                        owner.absence_status = status
                        owner.absence_message = message_text or msg.text[:100]

            # ===== SESSION CLOSED — медленные LLM-вызовы вне сессии =====

            # Urgent notification (LLM)
            if not msg.out and msg.text:
                _urgent_enabled = False
                _urgency_provider = None
                async with get_session() as _qs:
                    _o = await get_or_create_user(_qs, owner_telegram_id)
                    if _o.settings.urgent_notify_enabled:
                        from src.llm.router import build_provider

                        _urgency_provider = await build_provider(_qs, _o)
                        _urgent_enabled = True
                if _urgent_enabled and _urgency_provider:
                    from src.core.urgency_classifier import classify_urgency

                    urgency = await classify_urgency(
                        msg.text, provider=_urgency_provider, sender_name=sender_name
                    )
                    if urgency == "urgent":
                        sender_name_local = sender_name or str(peer_id)
                        await notifier.notify(
                            f"🔴 <b>СРОЧНОЕ от {sender_name_local}!</b>\n\n"
                            f"<i>{msg.text[:300]}</i>"
                        )
                    else:
                        # для отладки: логируем что LLM сказал про сообщение
                        logger.debug(
                            "Message from %s classified as %s: %s",
                            sender_name,
                            urgency,
                            msg.text[:80],
                        )

            # Draft suggestion (LLM)
            if not msg.out and msg.text:
                try:
                    _draft_sender = await event.get_sender()
                    _sender_is_bot = bool(getattr(_draft_sender, "bot", False))
                except Exception:
                    _sender_is_bot = False
                if not _sender_is_bot:
                    async with get_session() as _ds:
                        _ds_owner = await get_or_create_user(_ds, owner_telegram_id)
                        from src.core.draft_suggester import (
                            should_suggest,
                            suggest_draft,
                        )
                        from src.bot.handlers.draft_actions import (
                            draft_keyboard,
                            store_draft,
                        )
                        from src.core.text_sanitizer import sanitize_html
                        from src.llm.router import build_provider as _build_provider

                        _provider = await _build_provider(_ds, _ds_owner)

                        if await should_suggest(
                            _ds_owner.settings,
                            _ds_owner.id,
                            msg.text,
                            provider=_provider,
                        ):
                            if _provider:
                                from src.db.repo import (
                                    fetch_chat_messages,
                                    get_contact,
                                )

                                contact = await get_contact(_ds, _ds_owner, peer_id)
                                if contact:
                                    recent = await fetch_chat_messages(
                                        _ds,
                                        _ds_owner,
                                        peer_id,
                                        limit=10,
                                    )
                                    draft = await suggest_draft(
                                        _provider,
                                        _ds_owner.id,
                                        peer_id,
                                        contact,
                                        msg.text,
                                        sender_name or str(peer_id),
                                        recent,
                                    )
                                    if draft:
                                        draft_hash = store_draft(draft)
                                        safe_draft = sanitize_html(draft)[:400]
                                        await notifier.notify(
                                            f"💬 <b>{sender_name or peer_id}:</b>"
                                            f" <i>{msg.text[:200]}</i>\n\n"
                                            f"→ <b>Черновик:</b> {safe_draft}",
                                            reply_markup=draft_keyboard(
                                                peer_id, draft_hash
                                            ),
                                        )
        except Exception:
            logger.exception("mirror handler failed")

    client.add_event_handler(on_message, events.NewMessage())
    logger.info("Mirror handler attached for user %s", owner_telegram_id)
