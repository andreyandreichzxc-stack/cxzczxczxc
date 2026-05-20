import logging

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import CallbackQuery, InlineKeyboardButton, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder
from sqlalchemy import or_, select

from src.bot.filters import OwnerOnly
from src.core.contacts.chat_service import load_chat
from src.core.contacts.contact_resolver import resolve
from src.core.actions.indexer import index_chat
from src.core.actions.vector_store import vector_store
from src.db.models import Message as DBMessage
from src.db.repo import (
    FtsHit,
    cross_chat_search,
    get_contact,
    get_or_create_user,
)
from src.db.session import get_session
from src.llm.router import build_provider
from src.userbot.manager import UserbotManager


logger = logging.getLogger(__name__)
router = Router(name="search")
router.message.filter(OwnerOnly())
router.callback_query.filter(OwnerOnly())


def _result_keyboard(peer_id: int, message_id: int):
    kb = InlineKeyboardBuilder()
    kb.row(
        InlineKeyboardButton(
            text="➡ Переслать мне", callback_data=f"search:fwd:{peer_id}:{message_id}"
        ),
    )
    return kb.as_markup()


@router.message(Command("index"))
async def cmd_index(
    message: Message, command: CommandObject, userbot_manager: UserbotManager
) -> None:
    client = userbot_manager.get_client(message.from_user.id)
    if client is None:
        await message.answer("Сначала /login.")
        return
    query = (command.args or "").strip()
    if not query:
        await message.answer("Использование: <code>/index имя контакта</code>")
        return

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
    candidates = await resolve(client, owner, query)
    if not candidates:
        await message.answer("Не нашёл контакт. Попробуй /sync.")
        return
    if len(candidates) > 1 and candidates[0].score < 90:
        kb = InlineKeyboardBuilder()
        for c in candidates:
            kb.row(
                InlineKeyboardButton(
                    text=f"{c.label()} · {c.score}",
                    callback_data=f"search:idx:{c.peer_id}",
                )
            )
        kb.row(InlineKeyboardButton(text="❌ Отмена", callback_data="search:cancel:0"))
        await message.answer("Кого индексируем?", reply_markup=kb.as_markup())
        return

    await _do_index(message, candidates[0].peer_id, userbot_manager)


@router.callback_query(F.data.startswith("search:idx:"))
async def cb_idx_pick(callback: CallbackQuery, userbot_manager: UserbotManager) -> None:
    peer_id = int(callback.data.split(":")[2])
    if callback.message:
        await callback.message.edit_text("⏳ Индексирую...")
    await _do_index(
        callback.message, peer_id, userbot_manager, telegram_id=callback.from_user.id
    )
    await callback.answer()


async def _do_index(
    message_or_msg,
    peer_id: int,
    userbot_manager: UserbotManager,
    telegram_id: int | None = None,
) -> None:
    tg_id = telegram_id or message_or_msg.from_user.id
    client = userbot_manager.get_client(tg_id)
    if client is None:
        await message_or_msg.answer("Нет активного userbot. /login.")
        return

    # сначала подтянуть до 500 сообщений в БД
    await load_chat(client, tg_id, peer_id, limit=500, transcribe=True, parse_docs=True)

    async with get_session() as session:
        owner = await get_or_create_user(session, tg_id)
        contact = await get_contact(session, owner, peer_id)
        provider = await build_provider(session, owner)

    if not contact or not provider:
        await message_or_msg.answer("Контакт или LLM-ключ не найдены.")
        return

    n = await index_chat(provider, owner, contact)
    await message_or_msg.answer(
        f"✅ Проиндексировано <b>{n}</b> сообщений в чате с {contact.display_name}."
    )


@router.callback_query(F.data == "search:cancel:0")
async def cb_cancel(callback: CallbackQuery) -> None:
    if callback.message:
        await callback.message.edit_text("Отменено.")
    await callback.answer()


@router.message(Command("search"))
async def cmd_search(
    message: Message, command: CommandObject, userbot_manager: UserbotManager
) -> None:
    query = (command.args or "").strip()
    if not query:
        await message.answer("Использование: <code>/search текст запроса</code>")
        return

    async with get_session() as session:
        owner = await get_or_create_user(session, message.from_user.id)
        provider = await build_provider(session, owner)

    # Шаг 1: FTS-поиск по всем чатам (кросс-чатовый)
    async with get_session() as session:
        grouped = await cross_chat_search(session, owner, query, limit=30)

    # Если FTS ничего не дал — fallback на векторный / LIKE
    fts_hits: list[FtsHit] = []
    for peer_hits in grouped.values():
        fts_hits.extend(peer_hits)
    fts_hits.sort(key=lambda h: h.rank)

    if not fts_hits and provider is not None:
        # векторный fallback
        try:
            vec = await provider.embed(query)
            vec_hits = await vector_store.search(
                user_id=owner.id, embedding=vec, limit=8
            )
            for h in vec_hits:
                fts_hits.append(
                    FtsHit(
                        user_id=owner.id,
                        peer_id=h.peer_id,
                        message_id=h.message_id,
                        sender_name=h.peer_name,
                        snippet=h.text,
                        rank=h.score,
                        peer_name=h.peer_name,
                    )
                )
        except Exception:
            logger.exception("vector search failed")

    if not fts_hits:
        # LIKE fallback
        like = f"%{query}%"
        async with get_session() as session:
            result = await session.execute(
                select(DBMessage)
                .where(
                    DBMessage.user_id == owner.id,
                    or_(
                        DBMessage.text.ilike(like),
                        DBMessage.transcript.ilike(like),
                        DBMessage.extracted_text.ilike(like),
                    ),
                )
                .order_by(DBMessage.date.desc())
                .limit(8)
            )
            db_hits = list(result.scalars().all())
        for m in db_hits:
            fts_hits.append(
                FtsHit(
                    user_id=owner.id,
                    peer_id=m.peer_id,
                    message_id=m.message_id,
                    sender_name=m.sender_name,
                    snippet=(m.transcript or m.text or m.extracted_text or "")[:400],
                    rank=0.0,
                    peer_name=m.sender_name,
                    date=m.date,
                )
            )

    if not fts_hits:
        await message.answer(
            "Ничего не нашлось. Попробуй /index <контакт> для индексации."
        )
        return

    # Группировка по peer_id для кросс-чатового вывода
    from collections import defaultdict

    groups: dict[int, list[FtsHit]] = defaultdict(list)
    for hit in fts_hits:
        groups[hit.peer_id].append(hit)

    await message.answer(f"🔍 Результаты по «<i>{query}</i>»:")

    sent = 0
    for peer_id, hits in groups.items():
        peer_label = hits[0].peer_name or str(peer_id)
        await message.answer(f"📁 {peer_label} ({len(hits)} совпадений)")
        for hit in hits[:5]:  # не более 5 на чат
            if sent >= 8:
                break
            date_str = hit.date.strftime("%d.%m.%Y") if hit.date else ""
            body = hit.snippet[:200]
            line = f"• «{body}» — {date_str}" if date_str else f"• «{body}»"
            await message.answer(
                line, reply_markup=_result_keyboard(hit.peer_id, hit.message_id)
            )
            sent += 1
        if sent >= 8:
            break


@router.callback_query(F.data.startswith("search:fwd:"))
async def cb_forward(callback: CallbackQuery, userbot_manager: UserbotManager) -> None:
    parts = callback.data.split(":")
    peer_id = int(parts[2])
    msg_id = int(parts[3])
    client = userbot_manager.get_client(callback.from_user.id)
    if client is None:
        await callback.answer("Нет userbot, /login", show_alert=True)
        return
    try:
        entity = await client.get_entity(peer_id)
        await client.forward_messages("me", msg_id, entity)
    except Exception:
        logger.exception("forward failed")
        await callback.answer("Не удалось переслать", show_alert=True)
        return
    await callback.answer("Переслано в Saved Messages")
