"""Авто-синхронизация контактов и архивного статуса.Настраивается в /settings."""

import asyncio
import logging

from src.config import settings
from src.db.repo import get_or_create_user
from src.db.session import get_session
from src.userbot.dialogs import sync_dialogs


logger = logging.getLogger(__name__)


DEFAULT_SYNC_INTERVAL_SEC = 7200


from src.core.infra.task_manager import task_manager


@task_manager.task("auto-sync")
async def auto_sync_loop() -> None:
    from src.userbot import get_active_telethon_client

    while True:
        try:
            async with get_session() as session:
                owner = await get_or_create_user(session, settings.owner_telegram_id)
                enabled = owner.settings.auto_sync_enabled
                interval_sec = max(
                    30, getattr(owner.settings, "auto_sync_interval_sec", 7200)
                )

            if not enabled:
                await asyncio.sleep(settings.auto_sync_fallback_sec)
                continue

            client = get_active_telethon_client(settings.owner_telegram_id)
            if client is not None:
                async with get_session() as session:
                    owner = await get_or_create_user(
                        session, settings.owner_telegram_id
                    )
                stats = await sync_dialogs(client, owner, limit=500)
                logger.info("auto-sync done: %s", stats)

            await asyncio.sleep(interval_sec)
        except Exception:
            logger.exception("auto-sync tick failed")
            await asyncio.sleep(settings.auto_sync_fallback_sec)
