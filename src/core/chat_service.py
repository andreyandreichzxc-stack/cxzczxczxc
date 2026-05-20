import logging
from datetime import datetime, timezone
from pathlib import Path

from telethon import TelegramClient
from telethon.tl.custom import Message as TgMessage

from sqlalchemy import select, or_

from src.config import settings
from src.core.documents import extract_text, is_supported
from src.core.transcription import transcription_service
from src.db.models import Message, User, UserSettings
from src.db.repo import (
    fetch_chat_messages,
    get_api_key,
    get_or_create_user,
    upsert_message,
)
from src.db.session import get_session


logger = logging.getLogger(__name__)


def _media_dir(owner_telegram_id: int) -> Path:
    path = settings.data_dir / "media" / str(owner_telegram_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _classify(msg: TgMessage) -> str:
    if msg.voice:
        return "voice"
    if msg.audio:
        return "audio"
    if msg.document:
        return "document"
    if msg.photo:
        return "photo"
    if msg.text:
        return "text"
    return "other"


def _peer_id_from_message(msg: TgMessage) -> int:
    """Возвращает каноничный peer_id чата."""
    chat = msg.chat
    if chat is not None:
        return chat.id
    return msg.peer_id.user_id if hasattr(msg.peer_id, "user_id") else msg.chat_id


async def _sender_label(msg: TgMessage) -> str | None:
    sender = await msg.get_sender() if msg.sender_id else None
    if sender is None:
        return None
    parts = [getattr(sender, "first_name", None), getattr(sender, "last_name", None)]
    name = " ".join(p for p in parts if p).strip()
    if name:
        return name
    return getattr(sender, "username", None) or str(sender.id)


async def _process_one(
    client: TelegramClient,
    owner: User,
    msg: TgMessage,
    *,
    media_root: Path,
    transcribe: bool,
    parse_docs: bool,
    openai_key: str | None,
    gemini_key: str | None,
    mistral_key: str | None,
    transcription_mode: str,
    api_provider: str = "openai",
) -> None:
    kind = _classify(msg)
    peer_id = _peer_id_from_message(msg)
    sender_name = await _sender_label(msg)
    text = msg.text or msg.message or None
    transcript: str | None = None
    extracted: str | None = None
    media_path: str | None = None

    if kind in {"voice", "audio"} and transcribe:
        try:
            target = media_root / f"{peer_id}_{msg.id}.ogg"
            await msg.download_media(file=str(target))
            media_path = str(target)
            file_id = str(getattr(msg.file, "id", None) or f"{peer_id}:{msg.id}")
            transcript = await transcription_service.transcribe(
                target,
                file_id=file_id,
                mode=transcription_mode,
                openai_key=openai_key,
                gemini_key=gemini_key,
                mistral_key=mistral_key,
                api_provider=api_provider,
            )
            # Транскрипция готова — файл больше не нужен
            try:
                target.unlink(missing_ok=True)
                media_path = None
            except Exception:
                logger.debug("cleanup voice file failed: %s", target, exc_info=True)
        except Exception:
            logger.exception("transcription failed for msg %s", msg.id)

    elif kind == "document" and parse_docs:
        filename = getattr(msg.file, "name", None) or f"{msg.id}.bin"
        if is_supported(filename):
            try:
                target = media_root / f"{peer_id}_{msg.id}_{filename}"
                await msg.download_media(file=str(target))
                media_path = str(target)
                extracted = await extract_text(target)
                # Текст извлечён — документ больше не нужен
                try:
                    target.unlink(missing_ok=True)
                    media_path = None
                except Exception:
                    logger.debug("cleanup doc file failed: %s", target, exc_info=True)
            except Exception:
                logger.exception("doc parse failed for msg %s", msg.id)

    async with get_session() as session:
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
            transcript=transcript,
            media_path=media_path,
            extracted_text=extracted,
        )


async def _last_cached_message_id(owner_id: int, peer_id: int) -> int:
    async with get_session() as session:
        result = await session.execute(
            select(Message.message_id)
            .where(Message.user_id == owner_id, Message.peer_id == peer_id)
            .order_by(Message.message_id.desc())
            .limit(1)
        )
        row = result.scalar_one_or_none()
        return int(row or 0)


async def _cached_count(owner_id: int, peer_id: int) -> int:
    from sqlalchemy import func

    async with get_session() as session:
        result = await session.execute(
            select(func.count())
            .select_from(Message)
            .where(Message.user_id == owner_id, Message.peer_id == peer_id)
        )
        return int(result.scalar_one() or 0)


async def load_chat(
    client: TelegramClient,
    owner_telegram_id: int,
    peer_id: int,
    *,
    limit: int = 50,
    transcribe: bool = True,
    parse_docs: bool = False,
    incremental: bool = True,
) -> list[Message]:
    # incremental: если в БД достаточно сообщений, тянем только новее последнего
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_telegram_id)
        s: UserSettings = owner.settings
        openai_key = await get_api_key(session, owner, "openai")
        gemini_key = await get_api_key(session, owner, "gemini")
        mistral_key = await get_api_key(session, owner, "mistral")
        transcription_mode = s.transcription_mode
        api_provider = getattr(s, "transcription_api_provider", "openai")

    media_root = _media_dir(owner_telegram_id)
    entity = await client.get_entity(peer_id)

    iter_kwargs = {"limit": limit}
    if incremental:
        cached_total = await _cached_count(owner.id, peer_id)
        if cached_total >= limit:
            last_id = await _last_cached_message_id(owner.id, peer_id)
            if last_id:
                iter_kwargs = {"min_id": last_id, "limit": None}

    async for msg in client.iter_messages(entity, **iter_kwargs):
        await _process_one(
            client,
            owner,
            msg,
            media_root=media_root,
            transcribe=transcribe,
            parse_docs=parse_docs,
            openai_key=openai_key,
            gemini_key=gemini_key,
            mistral_key=mistral_key,
            transcription_mode=transcription_mode,
            api_provider=api_provider,
        )

    if transcribe:
        await _backfill_transcripts(
            client,
            owner.id,
            peer_id,
            limit=limit,
            media_root=media_root,
            openai_key=openai_key,
            gemini_key=gemini_key,
            mistral_key=mistral_key,
            transcription_mode=transcription_mode,
            api_provider=api_provider,
        )

    async with get_session() as session:
        owner = await get_or_create_user(session, owner_telegram_id)
        return await fetch_chat_messages(session, owner, peer_id, limit=limit)


async def _backfill_transcripts(
    client: TelegramClient,
    owner_id: int,
    peer_id: int,
    *,
    limit: int,
    media_root: Path,
    openai_key: str | None,
    gemini_key: str | None,
    mistral_key: str | None,
    transcription_mode: str,
    api_provider: str = "openai",
) -> None:
    # mirror кладёт voice/audio без transcript — здесь догоняем транскрипцию ленически
    async with get_session() as session:
        result = await session.execute(
            select(Message)
            .where(
                Message.user_id == owner_id,
                Message.peer_id == peer_id,
                Message.kind.in_(("voice", "audio")),
                Message.transcript.is_(None),
            )
            .order_by(Message.date.desc())
            .limit(limit)
        )
        pending = list(result.scalars().all())

    if not pending:
        return

    for m in pending:
        try:
            tg_msg = await client.get_messages(peer_id, ids=m.message_id)
            if tg_msg is None:
                continue
            target = media_root / f"{peer_id}_{m.message_id}.ogg"
            await tg_msg.download_media(file=str(target))
            file_id = str(
                getattr(tg_msg.file, "id", None) or f"{peer_id}:{m.message_id}"
            )
            transcript = await transcription_service.transcribe(
                target,
                file_id=file_id,
                mode=transcription_mode,
                openai_key=openai_key,
                gemini_key=gemini_key,
                mistral_key=mistral_key,
                api_provider=api_provider,
            )
            # Транскрипция готова — удаляем временный файл
            try:
                target.unlink(missing_ok=True)
            except Exception:
                logger.debug("backfill cleanup file failed: %s", target, exc_info=True)
        except Exception:
            logger.exception(
                "backfill transcript failed for msg %s in peer %s",
                m.message_id,
                peer_id,
            )
            continue

        if not transcript:
            continue
        async with get_session() as session:
            await upsert_message(
                session,
                user_id=owner_id,
                peer_id=peer_id,
                message_id=m.message_id,
                sender_id=m.sender_id,
                sender_name=m.sender_name,
                is_outgoing=m.is_outgoing,
                date=m.date,
                kind=m.kind,
                text=m.text,
                transcript=transcript,
                media_path=None,
                extracted_text=m.extracted_text,
            )


def message_to_text(m: Message) -> str:
    """Превращает Message в строку для LLM-промта."""
    body = m.transcript or m.text or m.extracted_text or f"[{m.kind}]"
    who = "Я" if m.is_outgoing else (m.sender_name or "Они")
    when = m.date.strftime("%Y-%m-%d %H:%M")
    return f"[{when}] {who}: {body}"


def messages_to_transcript(messages: list[Message]) -> str:
    return "\n".join(message_to_text(m) for m in messages)


async def sweep_orphaned_media() -> int:
    """Удаляет осиротевшие .ogg/.pdf/.docx в data/media/.
    Файл считается осиротевшим, если в БД у этого сообщения уже есть транскрипт.
    Возвращает количество удалённых файлов."""
    from sqlalchemy import or_

    from src.db.models import Message
    from src.db.session import get_session

    media_root = settings.data_dir / "media"
    if not media_root.exists():
        return 0

    deleted = 0
    # Собираем все message_id у которых есть транскрипт
    async with get_session() as session:
        result = await session.execute(
            select(Message.message_id, Message.peer_id)
            .where(
                or_(
                    Message.transcript.isnot(None),
                    Message.extracted_text.isnot(None),
                )
            )
            .group_by(Message.message_id, Message.peer_id)
        )
        transcribed_ids = {(r.message_id, r.peer_id) for r in result.all()}

    # Удаляем .ogg файлы
    for ogg in media_root.rglob("*.ogg"):
        try:
            name = ogg.stem  # формат: "{peer_id}_{message_id}"
            parts = name.split("_", 1)
            if len(parts) == 2:
                peer_id = int(parts[0])
                msg_id = int(parts[1])
                if (msg_id, peer_id) in transcribed_ids:
                    ogg.unlink(missing_ok=True)
                    deleted += 1
        except (ValueError, Exception):
            pass

    # Удаляем .pdf/.docx файлы (формат: "{peer_id}_{message_id}_{filename}")
    for ext in ("*.pdf", "*.docx"):
        for doc in media_root.rglob(ext):
            try:
                name = doc.stem
                parts = name.split("_", 2)
                if len(parts) >= 2:
                    peer_id = int(parts[0])
                    msg_id = int(parts[1])
                    if (msg_id, peer_id) in transcribed_ids:
                        doc.unlink(missing_ok=True)
                        deleted += 1
            except (ValueError, Exception):
                pass

    if deleted:
        logger.info("Swept %d orphaned media files", deleted)
    return deleted
