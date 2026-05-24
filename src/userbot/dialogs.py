"""Утилиты для работы с диалогами Telethon."""

import logging

from sqlalchemy import select as sa_select
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
        if getattr(entity, "username", None):
            return f"@{entity.username}"
        return f"User {entity.id}"
    title = getattr(entity, "title", None)
    return title or f"Chat {getattr(entity, 'id', '')}"


async def sync_dialogs(
    client: TelegramClient,
    owner: User,
    *,
    limit: int = 200,
    progress_callback=None,
) -> dict[str, int]:
    """Синхронизировать ВСЕ диалоги (архивные тоже).

    progress_callback — опциональная async-функция вида (current, total, peer_name) -> None.
    """
    stats = {
        "users": 0,
        "bots": 0,
        "chats": 0,
        "channels": 0,
        "archived": 0,
        "removed": 0,
    }
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
        # Сохранить папки в БД
        async with get_session() as session:
            owner = await session.merge(owner)
            await upsert_folders(session, owner, folders_data)
    except Exception as e:
        logger.warning("Failed to fetch folders: %s", e)
        peer_to_folder = {}

    # Собрать все диалоги в список (для прогресса и подсчёта общего числа)
    all_dialogs: list[tuple] = []
    for archived_pass in (False, True):
        async for dialog in client.iter_dialogs(limit=limit, archived=archived_pass):
            all_dialogs.append((dialog, archived_pass))

    total = len(all_dialogs)

    async with get_session() as session:
        owner = await session.merge(owner)
        active_peers: set[int] = set()

        for idx, (dialog, archived_pass) in enumerate(all_dialogs):
            entity = dialog.entity
            kind = _entity_kind(entity)
            is_bot = (
                bool(getattr(entity, "bot", False))
                if isinstance(entity, TgUser)
                else False
            )
            is_archived = bool(getattr(dialog, "archived", archived_pass))
            peer_id = entity.id
            active_peers.add(peer_id)
            await upsert_contact(
                session,
                owner,
                peer_id=peer_id,
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
            elif kind == "user" and is_bot:
                stats["bots"] += 1
            elif kind == "user":
                stats["users"] += 1
            elif kind == "chat":
                stats["chats"] += 1
            else:
                stats["channels"] += 1

            if progress_callback:
                await progress_callback(idx + 1, total, _entity_display_name(entity))

        # Очистка: удалить контакты, которых больше нет в диалогах Telegram
        if active_peers and limit >= 500:
            from sqlalchemy import delete as sa_delete
            from src.db.models import Contact

            result = await session.execute(
                sa_select(Contact).where(Contact.user_id == owner.id)
            )
            all_contacts = result.scalars().all()
            stale = [c.peer_id for c in all_contacts if c.peer_id not in active_peers]
            if stale:
                await session.execute(
                    sa_delete(Contact).where(
                        Contact.user_id == owner.id,
                        Contact.peer_id.in_(stale),
                    )
                )
                stats["removed"] = len(stale)
                logger.info(
                    "Removed %d stale contacts not in Telegram dialogs", len(stale)
                )
    return stats


async def sync_dialogs_with_options(
    client: TelegramClient,
    owner: User,
    *,
    include_private: bool = True,
    include_groups: bool = False,
    include_archived: bool = False,
    folder_names: list[str] | None = None,
    limit: int = 500,
    progress_callback=None,
) -> dict[str, int]:
    """Синхронизировать диалоги с фильтрацией.

    progress_callback — опциональная async-функция вида (current, total, peer_name) -> None.

    Возвращает: {"contacts": N, "synced": M, "skipped": K, "messages": P}
    """
    stats = {"contacts": 0, "synced": 0, "skipped": 0, "messages": 0, "removed": 0}

    # --- Папки пользователя Telegram ---
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
        async with get_session() as session:
            owner = await session.merge(owner)
            await upsert_folders(session, owner, folders_data)
    except Exception as e:
        logger.warning("Failed to fetch folders: %s", e)
        peer_to_folder = {}

    folder_set: set[str] | None = (
        set(folder_names) if folder_names is not None else None
    )

    # --- Сбор диалогов с фильтрацией ---
    collected: list = []
    passes = [False]
    if include_archived:
        passes.append(True)

    for archived in passes:
        async for dialog in client.iter_dialogs(limit=limit, archived=archived):
            entity = dialog.entity
            kind = _entity_kind(entity)
            peer_id = entity.id

            # Фильтр по папкам (если выбран) — переопределяет фильтр по типу
            if folder_set is not None:
                dialog_folders_str = peer_to_folder.get(peer_id, "")
                dlg_folders = (
                    set(dialog_folders_str.split(",")) if dialog_folders_str else set()
                )
                if not dlg_folders.intersection(folder_set):
                    stats["skipped"] += 1
                    continue
            else:
                # Фильтр по типу чата
                is_user = kind == "user"
                is_group_or_channel = kind in ("chat", "channel")
                if not include_private and is_user:
                    stats["skipped"] += 1
                    continue
                if not include_groups and is_group_or_channel:
                    stats["skipped"] += 1
                    continue

            collected.append(dialog)

    stats["contacts"] = len(collected)
    total = len(collected)

    # --- Обработка диалогов ---
    async with get_session() as session:
        owner = await session.merge(owner)
        active_peers: set[int] = set()

        for idx, dialog in enumerate(collected):
            entity = dialog.entity
            kind = _entity_kind(entity)
            is_bot = (
                bool(getattr(entity, "bot", False))
                if isinstance(entity, TgUser)
                else False
            )
            is_archived = bool(getattr(dialog, "archived", False))
            peer_id = entity.id
            active_peers.add(peer_id)

            await upsert_contact(
                session,
                owner,
                peer_id=peer_id,
                peer_kind=kind,
                is_bot=is_bot,
                is_archived=is_archived,
                display_name=_entity_display_name(entity),
                username=getattr(entity, "username", None),
                phone=getattr(entity, "phone", None),
                folder_names=peer_to_folder.get(entity.id),
            )

            stats["synced"] += 1

            if progress_callback:
                await progress_callback(idx + 1, total, _entity_display_name(entity))

        # Очистка неактуальных контактов (только при полном обходе)
        if total >= 500:
            from sqlalchemy import delete as sa_delete
            from src.db.models import Contact

            result = await session.execute(
                sa_select(Contact).where(Contact.user_id == owner.id)
            )
            all_contacts = result.scalars().all()
            stale = [c.peer_id for c in all_contacts if c.peer_id not in active_peers]
            if stale:
                await session.execute(
                    sa_delete(Contact).where(
                        Contact.user_id == owner.id,
                        Contact.peer_id.in_(stale),
                    )
                )
                stats["removed"] = len(stale)
                logger.info("Removed %d stale contacts", len(stale))

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
