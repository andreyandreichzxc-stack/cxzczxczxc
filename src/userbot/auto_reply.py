"""Авто-ответ оффлайн. Жёсткие правила: только входящие в ЛС от людей (не боты),
только если включено и владелец действительно оффлайн (или статус скрыт),
один ответ на контакт раз в COOLDOWN_MINUTES."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from telethon import TelegramClient, events
from telethon.tl.custom import Message as TgMessage
from telethon.tl.types import (
    User as TgUser,
    UserStatusOffline,
    UserStatusOnline,
)

from src.config import settings as app_settings
from src.core.chat_service import load_chat, message_to_text
from src.core.notifier import notifier
from src.core.style_profile import style_profile_as_prompt_hint
from src.core.timeutil import now_in_tz
from src.core.vector_store import vector_store
from src.db.models import AutoReplyLog, User
from src.db.repo import (
    add_auto_reply_log,
    get_contact,
    get_or_create_user,
    list_memories,
    upsert_contact,
)
from src.db.session import get_session
from src.llm.base import ChatMessage
from src.llm.router import build_provider


logger = logging.getLogger(__name__)


COOLDOWN_MINUTES = 30
CONTEXT_LIMIT = 100


AUTO_REPLY_SYSTEM_BASE = (
    "Ты отвечаешь от моего имени, потому что я сейчас оффлайн.\n"
    "Определи по контексту переписки мои отношения с собеседником и выбери ТОН ответа:\n"
    "- Друг/подруга, близкий человек, тёплое общение: ОЧЕНЬ тёплый, ласковый, на «ты», "
    "используй уменьшительно-ласкательные формы имени (Настя→Настюша, Саша→Сашуля, "
    "Катя→Катюша, Дима→Димочка, Лена→Леночка), с сердечками/смайликами. "
    "Напиши что-то вроде «солнышко, я пока не у телефона, как только освобожусь — сразу отвечу!»\n"
    "- Коллега/рабочий контакт: вежливо, по-деловому, без фамильярности.\n"
    "- Незнакомец/малознакомый: холодно, сухо, одной фразой. Без смайликов.\n"
    "- Если характер переписки неясен — нейтрально.\n\n"
    "ПРАВИЛА:\n"
    "Не пиши длиннее 1–3 коротких предложений.\n"
    "Если просят что-то конкретное — не обещай за меня, скажи: «передам, сейчас занят».\n"
    "Никогда не выдавай, что ты ИИ. Не упоминай ассистента, бота.\n"
    "Не исользуй «уважаемый», «господин» — я так не общаюсь.\n"
    "- Если я СПЛЮ (ночное время): честно скажи что я сплю, с эмодзи 😴💤🌙🛏️. Не используй «занят»/«не у телефона»."
)


async def _check_and_track_offline(
    client: TelegramClient, session: AsyncSession, owner: User
) -> bool:
    try:
        me = await client.get_me()
        status = getattr(me, "status", None)
        if isinstance(status, UserStatusOnline):
            owner.last_seen_online = datetime.utcnow()
            # Сброс sleeping статуса — владелец онлайн
            if owner.absence_status == "sleeping":
                owner.absence_status = None
                owner.absence_message = None
            await session.commit()
            return False
        if isinstance(status, UserStatusOffline):
            now = datetime.utcnow()
            last_seen = owner.last_seen_online
            if last_seen is None or (now - last_seen) > timedelta(minutes=10):
                # Sleep detection — определяем, не спит ли владелец
                tz_name = owner.settings.timezone if owner.settings else "UTC"
                local_now = now_in_tz(tz_name)
                hour = local_now.hour
                is_night = hour >= 22 or hour < 8

                if is_night:
                    if last_seen is not None:
                        offline_minutes = (now - last_seen).total_seconds() / 60
                        if offline_minutes > 30 and owner.absence_status != "sleeping":
                            owner.absence_status = "sleeping"
                            owner.absence_message = (
                                f"Спит с {local_now.strftime('%H:%M')}"
                            )
                            await session.commit()
                else:
                    # Дневное время — сброс sleeping статуса
                    if owner.absence_status == "sleeping":
                        owner.absence_status = None
                        owner.absence_message = None
                        await session.commit()

                return True
            return False
        return True
    except Exception:
        logger.exception("get_me failed in _check_and_track_offline")
        return False


async def _recently_replied(owner_telegram_id: int, peer_id: int) -> bool:
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_telegram_id)
        cooldown = getattr(owner.settings, "auto_reply_cooldown_min", None) or 30
        threshold = datetime.utcnow() - timedelta(minutes=cooldown)
        result = await session.execute(
            select(AutoReplyLog)
            .where(
                AutoReplyLog.user_id == owner.id,
                AutoReplyLog.peer_id == peer_id,
                AutoReplyLog.created_at >= threshold,
            )
            .limit(1)
        )
        return result.scalar_one_or_none() is not None


async def _build_reply_text(
    owner_telegram_id: int,
    peer_id: int,
    sender_name: str,
    incoming_text: str,
) -> str | None:
    memory_context = ""
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_telegram_id)
        provider = await build_provider(session, owner)
        contact = await get_contact(session, owner, peer_id)

        # Векторный поиск релевантных фактов
        relevant_facts = []
        try:
            if provider:
                query_vec = await provider.embed(incoming_text[:300])
                hits = await vector_store.search_similar_memories(
                    user_id=owner.id,
                    embedding=query_vec,
                    limit=10,
                    threshold=0.7,
                    contact_id=peer_id,
                )
                for h in hits:
                    # Скоринг: 0.5*sim + 0.3*conf + 0.2*importance
                    sim = h.get("score", 0.5)
                    conf = float(h.get("confidence", 0.5))
                    imp = float(h.get("importance", 0.5))
                    score = 0.5 * sim + 0.3 * conf + 0.2 * imp
                    relevant_facts.append((score, h.get("fact", "")))
                relevant_facts.sort(key=lambda x: x[0], reverse=True)
        except Exception:
            logger.warning("Vector memory search failed, using fallback")
            # fallback к старому поведению
            try:
                memories = await list_memories(session, owner, contact_id=peer_id)
                if memories:
                    relevant_facts = [(0.5, m.fact) for m in memories[-5:]]
            except Exception:
                pass

        if relevant_facts:
            memory_lines = [f"- {f}" for _, f in relevant_facts[:5]]
            memory_context = "Релевантные факты из памяти:\n" + "\n".join(memory_lines)

        # Contact Archetype — вычисляем если ещё не задан
        if contact and contact.archetype is None:
            from src.core.contact_archetypes import classify_contact

            archetype = await classify_contact(owner_telegram_id, peer_id)
            if archetype:
                contact.archetype = archetype
                await session.commit()

        heavy = owner.settings.use_heavy_model
        global_profile = owner.global_style_profile

    if provider is None:
        logger.warning("auto-reply: no LLM provider configured")
        return None

    # подгружаем контекст последних сообщений
    from src.userbot.manager import _MANAGER_SINGLETON  # локальный импорт

    client = (
        _MANAGER_SINGLETON.get_client(owner_telegram_id) if _MANAGER_SINGLETON else None
    )
    history_text = ""
    if client is not None:
        try:
            messages = await load_chat(
                client, owner_telegram_id, peer_id, limit=CONTEXT_LIMIT
            )
            history_text = "\n".join(
                message_to_text(m) for m in messages[-CONTEXT_LIMIT:]
            )
        except Exception:
            logger.exception("auto-reply: load_chat failed")

    style_hint = style_profile_as_prompt_hint(
        contact.style_profile if contact else None,
        global_profile,
    )
    system = AUTO_REPLY_SYSTEM_BASE
    if memory_context:
        system = system + "\n\n" + memory_context
    if owner.absence_status == "away":
        system += f"\n\nВАЖНО: Владелец сказал перед уходом: «{owner.absence_message}». Учти это в ответе. Он отсутствует."
    elif owner.absence_status == "soon_back":
        system += f"\n\nВладелец скоро вернётся: «{owner.absence_message}». Ответь обнадёживающе, он скоро будет."
    elif owner.absence_status == "sleeping":
        system += (
            f"\n\n🌙💤 Владелец СПИТ ({owner.absence_message}). "
            "Никаких «занят» или «не у телефона» — честно скажи что он спит. "
            "Используй эмодзи: 😴🛏️🌙💤🌌. Тон: заботливый, сонный. "
            "Пример: «Владелец сейчас спит сладким сном 😴💤 "
            "Как проснётся — обязательно ответит! 🌙»"
        )
    if style_hint:
        system = system + "\n" + style_hint

    # Архетип контакта (подсказка для тона)
    if contact and contact.archetype:
        from src.core.contact_archetypes import archetype_reply_hint

        hint = archetype_reply_hint(contact.archetype)
        if hint:
            system += hint

    user_prompt = (
        f"Собеседник: {sender_name}.\n"
        f"Контекст последних сообщений:\n{history_text}\n\n"
        f"Последнее входящее: {incoming_text}\n\n"
        "Сформируй ответ от моего имени."
    )
    try:
        return await provider.chat(
            [
                ChatMessage(role="system", content=system),
                ChatMessage(role="user", content=user_prompt),
            ],
            heavy=heavy,
        )
    except Exception:
        logger.exception("auto-reply: LLM call failed")
        return None


async def _make_handler(client: TelegramClient, owner_telegram_id: int):
    """Возвращает event handler, замкнутый на owner_telegram_id."""

    async def handler(event: events.NewMessage.Event) -> None:
        try:
            msg: TgMessage = event.message
            if msg.out:
                return
            sender = await event.get_sender()
            if not isinstance(sender, TgUser) or sender.bot:
                return  # только ЛС от человеков
            if not event.is_private:
                return

            async with get_session() as session:
                owner: User = await get_or_create_user(session, owner_telegram_id)
                if not owner.settings.auto_reply_enabled:
                    return
                # игнорируем архив, если опция включена
                existing = await get_contact(session, owner, sender.id)
                if (
                    owner.settings.ignore_archived
                    and existing is not None
                    and existing.is_archived
                ):
                    return
                # запомним контакт
                parts = [
                    getattr(sender, "first_name", None),
                    getattr(sender, "last_name", None),
                ]
                display = " ".join(p for p in parts if p).strip() or (
                    sender.username or str(sender.id)
                )
                await upsert_contact(
                    session,
                    owner,
                    peer_id=sender.id,
                    peer_kind="user",
                    is_bot=bool(getattr(sender, "bot", False)),
                    display_name=display,
                    username=getattr(sender, "username", None),
                    phone=getattr(sender, "phone", None),
                )

                # Folder filter: если monitor_only_selected_folders и контакт не в выбранных папках — не отвечаем
                if (
                    owner.settings.monitor_only_selected_folders
                    and owner.settings.monitored_folders
                ):
                    import json as _ar_json

                    monitored = _ar_json.loads(owner.settings.monitored_folders)
                    if monitored:
                        contact = existing  # уже загружен через get_contact выше
                        contact_folders = (
                            (contact.folder_names or "").split(",") if contact else []
                        )
                        contact_folders = [
                            f.strip() for f in contact_folders if f.strip()
                        ]
                        if not any(f in monitored for f in contact_folders):
                            return

                if not await _check_and_track_offline(client, session, owner):
                    return
            if await _recently_replied(owner_telegram_id, sender.id):
                return

            incoming_text = msg.text or msg.message or ""
            if not incoming_text.strip():
                return  # медиа без текста — не отвечаем автоматически

            # перечитаем настройки для режима/текста
            async with get_session() as session:
                owner = await get_or_create_user(session, owner_telegram_id)
                mode = owner.settings.auto_reply_mode
                static_text = owner.settings.auto_reply_text or ""

            if mode == "smart":
                reply = await _build_reply_text(
                    owner_telegram_id, sender.id, display, incoming_text
                )
                if not reply:
                    return
            else:  # static (default)
                reply = static_text.strip()
                if not reply:
                    return

            await event.respond(reply)

            # Обновить ConversationState
            async with get_session() as _ar_session:
                _ar_owner = await get_or_create_user(_ar_session, owner_telegram_id)
                from src.db.repo import upsert_conversation_state

                await upsert_conversation_state(
                    _ar_session,
                    _ar_owner,
                    sender.id,
                    status="active",
                    last_outgoing_at=datetime.utcnow(),
                    last_auto_reply_at=datetime.utcnow(),
                )

            async with get_session() as session:
                owner = await get_or_create_user(session, owner_telegram_id)
                await add_auto_reply_log(
                    session,
                    user_id=owner.id,
                    peer_id=sender.id,
                    peer_name=display,
                    incoming_text=incoming_text[:500],
                    reply_text=reply,
                )

            await notifier.notify(
                f"🤖 <b>Авто-ответ</b> для <b>{display}</b>\n\n"
                f"<i>Им:</i> {incoming_text[:200]}\n"
                f"<i>Я:</i> {reply}"
            )
        except Exception:
            logger.exception("auto-reply handler failed")

    return handler


async def generate_smart_reply(
    client: TelegramClient,
    owner_telegram_id: int,
    peer_id: int,
    sender_name: str,
    incoming_text: str,
) -> str | None:
    """Публичная обёртка для генерации умного авто-ответа.

    Вызывается из InboxManager или напрямую из других модулей.
    Возвращает сгенерированный текст или None."""
    try:
        return await _build_reply_text(
            owner_telegram_id=owner_telegram_id,
            peer_id=peer_id,
            sender_name=sender_name,
            incoming_text=incoming_text,
        )
    except Exception:
        logger.exception("generate_smart_reply failed")
        return None


def attach_auto_reply(client: TelegramClient, owner_telegram_id: int) -> None:
    _handler_cache = None

    async def _wrapper(event):
        nonlocal _handler_cache
        if _handler_cache is None:
            _handler_cache = await _make_handler(client, owner_telegram_id)
        await _handler_cache(event)

    client.add_event_handler(_wrapper, events.NewMessage(incoming=True))
    logger.info("Auto-reply handler attached for user %s", owner_telegram_id)
