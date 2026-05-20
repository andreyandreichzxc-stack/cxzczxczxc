"""Зеркало всех сообщений (входящих и исходящих) в БД и FTS5 в реальном времени.
Транскрипция голоса и парсинг документов — лениво в момент анализа."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from telethon import TelegramClient, events
from telethon.tl.custom import Message as TgMessage
from telethon.tl.types import User as TgUser

from src.core.notifier import notifier
from src.db.repo import (
    get_contact,
    get_or_create_user,
    upsert_contact,
    upsert_conversation_state,
    upsert_message,
)
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


async def _process_incoming_bg(
    owner_telegram_id: int,
    peer_id: int,
    sender_name: str,
    text: str,
) -> None:
    """Фоновая обработка входящего сообщения: InboxManager + notifier.

    Открывает собственную сессию БД, не роняет обработчик при ошибках.
    """
    from src.core.inbox_manager import InboxAction, process_incoming
    from src.db.repo import get_contact, get_or_create_user, upsert_conversation_state
    from src.llm.router import build_provider

    try:
        async with get_session() as _im_session:
            _im_owner = await get_or_create_user(_im_session, owner_telegram_id)
            _im_contact = await get_contact(_im_session, _im_owner, peer_id)
            _im_provider = await build_provider(_im_session, _im_owner)
            decision = await process_incoming(
                message_text=text,
                sender_name=sender_name,
                peer_id=peer_id,
                owner=_im_owner,
                contact=_im_contact,
                provider=_im_provider,
            )

            # Обновить ConversationState
            status = "active"
            if decision.action == InboxAction.QUEUE_FOR_DIGEST:
                status = "waiting_reply"
            await upsert_conversation_state(
                _im_session,
                _im_owner,
                peer_id,
                status=status,
                increment_unread=True,
                last_incoming_at=datetime.now(timezone.utc).replace(tzinfo=None),
            )

        # Применить решение (вне сессии)
        if decision.action == InboxAction.NOTIFY_URGENT:
            await notifier.notify(
                f"🔴 <b>СРОЧНОЕ от {sender_name}!</b>\n\n<i>{text[:300]}</i>"
            )
        elif decision.action == InboxAction.DRAFT_SUGGEST:
            from src.core.notification_queue import notification_queue

            await notification_queue.enqueue(
                topic="inbox",
                text=f"💬 <b>{sender_name}:</b> <i>{text[:200]}</i>\n\n→ Напиши ответ? /chat {sender_name}",
                priority=2,
                category="draft",
            )
        # SILENT_LOG / IGNORE — только сохранили в БД, ничего не делаем
    except Exception:
        logger.exception("Background inbox processing failed for peer %s", peer_id)


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
                    else datetime.now(timezone.utc).replace(tzinfo=None),
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

            # ===== InboxManager: тяжёлая обработка — в фон =====
            if not msg.out and msg.text:
                asyncio.create_task(
                    _process_incoming_bg(
                        owner_telegram_id=owner_telegram_id,
                        peer_id=peer_id,
                        sender_name=sender_name or str(peer_id),
                        text=msg.text,
                    )
                )
        except Exception:
            logger.exception("mirror handler failed")

    client.add_event_handler(on_message, events.NewMessage())
    logger.info("Mirror handler attached for user %s", owner_telegram_id)
