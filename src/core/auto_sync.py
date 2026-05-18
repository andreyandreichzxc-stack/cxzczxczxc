"""Раз в час обновляем кэш контактов и архивный статус. Страховка к live-handler'у
на UpdateFolderPeers — на случай пропусков во время даунтайма."""
import asyncio
import logging

from src.config import settings as app_settings
from src.db.repo import get_or_create_user
from src.db.session import get_session
from src.userbot.dialogs import sync_dialogs


logger = logging.getLogger(__name__)


AUTO_SYNC_SECONDS = 3600


async def auto_sync_loop() -> None:
    from src.userbot.manager import _MANAGER_SINGLETON  # отложенный импорт против цикла

    while True:
        try:
            manager = _MANAGER_SINGLETON
            if manager is not None:
                client = manager.get_client(app_settings.owner_telegram_id)
                if client is not None:
                    async with get_session() as session:
                        owner = await get_or_create_user(session, app_settings.owner_telegram_id)
                    stats = await sync_dialogs(client, owner, limit=500)
                    logger.info("auto-sync done: %s", stats)
        except Exception:
            logger.exception("auto-sync tick failed")
        await asyncio.sleep(AUTO_SYNC_SECONDS)
