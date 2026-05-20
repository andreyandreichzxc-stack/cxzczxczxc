"""Утилиты для работы с диалогами Telethon."""

import logging

from telethon import TelegramClient
from telethon.tl.types import Channel, Chat, User as TgUser

from src.core.contacts.chat_service import load_chat
from src.db.models import User
from src.db.repo import upsert_contact, upsert_folders
from src.db.session import get_session


logger = logging.getLogger(__name__)


def _entity_kind(entity: object) -> str:
    if isinstance(entity, TgUser):
        return "user"
    if isinstance(entity, Chat):
        return "chat"
    if isinstance(entity, Channel):
        return "channel"
    return "user"


def _entity_display_name(entity: object) -> str:
    if isinstance(entity, TgUser):
        parts = [
            getattr(entity, "first_name", None),
            getattr(entity, "last_name", None),
        ]
        name = " ".join(p for p in parts if p).strip()
        if name:
            return name
        return entity.username or str(entity.id)
    title = getattr(entity, "title", None)
    return title or str(getattr(entity, "id", ""))


async def sync_dialogs(
    client: TelegramClient, owner: User, *, limit: int = 200
) -> dict[str, int]:
    stats = {"users": 0, "bots": 0, "chats": 0, "channels": 0, "archived": 0}
    # Получить папки пользователя Telegram
    from telethon.tl.functions.messages import GetDialogFiltersRequest

    try:
        filters_result = await client(GetDialogFiltersRequest())
        peer_to_folder: dict[int, str] = {}
        folders_data = []
        for f in filters_result.filters:
            if hasattr(f, "title") and f.title:
                folders_data.append(
                    {
                        "telegram_folder_id": f.id if hasattr(f, "id") else 0,
                        "title": f.title,
                        "emoji": getattr(f, "emoticon", None),
                    }
                )
                # Маппинг пиров в папку
                if hasattr(f, "include_peers"):
                    for peer in f.include_peers:
                        try:
                            pid = (
                                getattr(peer, "user_id", None)
                                or getattr(peer, "chat_id", None)
                                or getattr(peer, "channel_id", None)
                            )
                            if pid:
                                existing = peer_to_folder.get(pid, "")
                                peer_to_folder[pid] = (
                                    (existing + "," + f.title) if existing else f.title
                                )
                        except Exception:
                            logger.debug(
                                "dialogs: folder sync entry skipped", exc_info=True
                            )
                            pass
        # Сохранить папки в БД
        async with get_session() as session:
            await upsert_folders(session, owner, folders_data)
    except Exception as e:
        logger.warning("Failed to fetch folders: %s", e)
        peer_to_folder = {}
    async with get_session() as session:
        # archived=True — это отдельная архивная папка, делаем два прохода
        for archived_pass in (False, True):
            async for dialog in client.iter_dialogs(
                limit=limit, archived=archived_pass
            ):
                entity = dialog.entity
                kind = _entity_kind(entity)
                is_bot = (
                    bool(getattr(entity, "bot", False))
                    if isinstance(entity, TgUser)
                    else False
                )
                is_archived = bool(getattr(dialog, "archived", archived_pass))
                await upsert_contact(
                    session,
                    owner,
                    peer_id=entity.id,
                    peer_kind=kind,
                    is_bot=is_bot,
                    is_archived=is_archived,
                    display_name=_entity_display_name(entity),
                    username=getattr(entity, "username", None),
                    phone=getattr(entity, "phone", None),
                    folder_names=peer_to_folder.get(entity.id),
                )
                if is_archived:
                    stats["archived"] += 1
                    continue
                if kind == "user" and is_bot:
                    stats["bots"] += 1
                elif kind == "user":
                    stats["users"] += 1
                elif kind == "chat":
                    stats["chats"] += 1
                else:
                    stats["channels"] += 1
    return stats


async def prefetch_recent_messages(
    client: TelegramClient,
    owner_telegram_id: int,
    *,
    top_n: int = 30,
    per_chat: int = 50,
    skip_bots: bool = True,
    skip_channels: bool = False,
) -> dict[str, int]:
    # один разовый прогон при /sync — заполняет БД и FTS5 для холодного старта
    stats = {"chats": 0, "messages": 0, "skipped": 0}
    async for dialog in client.iter_dialogs(limit=top_n * 3, archived=False):
        if stats["chats"] >= top_n:
            break
        entity = dialog.entity
        is_bot = isinstance(entity, TgUser) and bool(getattr(entity, "bot", False))
        if skip_bots and is_bot:
            stats["skipped"] += 1
            continue
        is_channel_only = isinstance(entity, Channel) and getattr(
            entity, "broadcast", False
        )
        if skip_channels and is_channel_only:
            stats["skipped"] += 1
            continue
        try:
            msgs = await load_chat(
                client,
                owner_telegram_id,
                entity.id,
                limit=per_chat,
                transcribe=False,
                parse_docs=False,
                incremental=True,
            )
            stats["chats"] += 1
            stats["messages"] += len(msgs)
        except Exception:
            logger.exception("prefetch failed for peer %s", entity.id)
            stats["skipped"] += 1
    return stats
