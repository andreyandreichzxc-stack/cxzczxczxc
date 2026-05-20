import asyncio
import logging
from dataclasses import dataclass, field

from telethon import TelegramClient
from telethon.errors import FloodWaitError
from telethon.sessions import StringSession

from src.config import parse_telethon_proxy, settings
from src.db.repo import load_telegram_session
from src.db.session import get_session


logger = logging.getLogger(__name__)


@dataclass
class PendingLogin:
    # Промежуточное состояние логина: между запросами кода и 2FA-пароля

    client: TelegramClient
    api_id: int
    api_hash: str
    phone: str | None = None
    phone_code_hash: str | None = None


_MANAGER_SINGLETON: "UserbotManager | None" = None


async def _retry_after_floodwait(
    telegram_id: int,
    session_string: str,
    api_id: int,
    api_hash: str,
    delay: float,
) -> None:
    """Ждёт FloodWait и переподключает клиента."""
    await asyncio.sleep(delay)
    logger.info("Retrying after FloodWait for %s", telegram_id)
    mgr = _MANAGER_SINGLETON
    if mgr is None:
        return
    client = TelegramClient(
        StringSession(session_string),
        api_id,
        api_hash,
        proxy=parse_telethon_proxy(settings.proxy_url),
    )
    try:
        await client.connect()
        if await client.is_user_authorized():
            mgr._clients[telegram_id] = client
            from src.userbot.auto_reply import attach_auto_reply
            from src.userbot.dialog_events import attach_dialog_event_handlers
            from src.userbot.mirror import attach_mirror

            attach_auto_reply(client, telegram_id)
            attach_dialog_event_handlers(client, telegram_id)
            attach_mirror(client, telegram_id)
            logger.info("FloodWait retry succeeded for %s", telegram_id)
        else:
            await client.disconnect()
            logger.warning("FloodWait retry: session expired for %s", telegram_id)
    except Exception:
        logger.exception("FloodWait retry failed for %s", telegram_id)
        if client.is_connected():
            await client.disconnect()


@dataclass
class UserbotManager:
    _clients: dict[int, TelegramClient] = field(default_factory=dict)
    _pending: dict[int, PendingLogin] = field(default_factory=dict)

    def __post_init__(self) -> None:
        global _MANAGER_SINGLETON
        _MANAGER_SINGLETON = self

    async def restore_all(self) -> None:
        async with get_session() as session:
            from src.db.models import User
            from sqlalchemy import select

            users = (await session.execute(select(User))).scalars().all()
            for user in users:
                creds = await load_telegram_session(session, user)
                if creds is None:
                    continue
                api_id, api_hash, session_string = creds
                client = TelegramClient(
                    StringSession(session_string),
                    api_id,
                    api_hash,
                    proxy=parse_telethon_proxy(settings.proxy_url),
                )
                try:
                    await client.connect()
                    if await client.is_user_authorized():
                        self._clients[user.telegram_id] = client
                        from src.userbot.auto_reply import attach_auto_reply
                        from src.userbot.dialog_events import (
                            attach_dialog_event_handlers,
                        )
                        from src.userbot.mirror import attach_mirror

                        attach_auto_reply(client, user.telegram_id)
                        attach_dialog_event_handlers(client, user.telegram_id)
                        attach_mirror(client, user.telegram_id)
                        logger.info(
                            "Restored Telethon client for user %s", user.telegram_id
                        )
                    else:
                        await client.disconnect()
                        logger.warning(
                            "Saved session for %s is not authorized anymore",
                            user.telegram_id,
                        )
                        from src.core.notification_queue import notification_queue

                        await notification_queue.enqueue(
                            topic=f"userbot:{user.telegram_id}",
                            text="🔐 Сессия Telegram протухла. Нужен повторный /login.",
                            priority=1,  # Notification.PRIORITY_HIGH
                        )
                except FloodWaitError as e:
                    logger.warning(
                        "FloodWait %ds for user %s, scheduling retry",
                        e.seconds,
                        user.telegram_id,
                    )
                    if client.is_connected():
                        await client.disconnect()
                    asyncio.create_task(
                        _retry_after_floodwait(
                            telegram_id=user.telegram_id,
                            session_string=session_string,
                            api_id=api_id,
                            api_hash=api_hash,
                            delay=e.seconds,
                        )
                    )
                except Exception:
                    logger.exception(
                        "Failed to restore client for user %s", user.telegram_id
                    )
                    if client.is_connected():
                        await client.disconnect()

    def get_client(self, telegram_id: int) -> TelegramClient | None:
        return self._clients.get(telegram_id)

    def register_client(self, telegram_id: int, client: TelegramClient) -> None:
        self._clients[telegram_id] = client
        from src.userbot.auto_reply import attach_auto_reply
        from src.userbot.dialog_events import attach_dialog_event_handlers
        from src.userbot.mirror import attach_mirror

        attach_auto_reply(client, telegram_id)
        attach_dialog_event_handlers(client, telegram_id)
        attach_mirror(client, telegram_id)

    async def remove_client(self, telegram_id: int) -> None:
        client = self._clients.pop(telegram_id, None)
        if client is not None:
            try:
                await client.log_out()
            except Exception:
                logger.exception("log_out failed")
            try:
                await client.disconnect()
            except Exception:
                logger.exception("userbot disconnect failed")

    def start_pending(
        self, telegram_id: int, api_id: int, api_hash: str
    ) -> PendingLogin:
        client = TelegramClient(
            StringSession(),
            api_id,
            api_hash,
            proxy=parse_telethon_proxy(settings.proxy_url),
        )
        pending = PendingLogin(client=client, api_id=api_id, api_hash=api_hash)
        self._pending[telegram_id] = pending
        return pending

    def get_pending(self, telegram_id: int) -> PendingLogin | None:
        return self._pending.get(telegram_id)

    async def cancel_pending(self, telegram_id: int) -> None:
        pending = self._pending.pop(telegram_id, None)
        if pending is not None:
            try:
                await pending.client.disconnect()
            except Exception:
                logger.exception("userbot disconnect failed")

    def clear_pending(self, telegram_id: int) -> PendingLogin | None:
        return self._pending.pop(telegram_id, None)
