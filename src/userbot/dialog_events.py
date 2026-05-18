"""Live-апдейт is_archived через Telethon UpdateFolderPeers (folder_id=1 — архив)."""
from __future__ import annotations

import logging

from telethon import TelegramClient, events
from telethon.tl.types import UpdateFolderPeers
from telethon.utils import get_peer_id

from src.db.repo import get_contact, get_or_create_user
from src.db.session import get_session


logger = logging.getLogger(__name__)


def attach_dialog_event_handlers(client: TelegramClient, owner_telegram_id: int) -> None:
    async def on_folder_peers(update: UpdateFolderPeers) -> None:
        try:
            async with get_session() as session:
                owner = await get_or_create_user(session, owner_telegram_id)
                touched = 0
                for fp in update.folder_peers:
                    try:
                        peer_id = get_peer_id(fp.peer)
                    except Exception:
                        continue
                    is_archived = (fp.folder_id == 1)
                    contact = await get_contact(session, owner, peer_id)
                    if contact is None:
                        continue
                    if contact.is_archived != is_archived:
                        contact.is_archived = is_archived
                        touched += 1
                if touched:
                    logger.info("UpdateFolderPeers: updated archive flag for %d contacts", touched)
        except Exception:
            logger.exception("on_folder_peers handler failed")

    client.add_event_handler(on_folder_peers, events.Raw(types=[UpdateFolderPeers]))
    logger.info("Dialog event handler attached for user %s", owner_telegram_id)
