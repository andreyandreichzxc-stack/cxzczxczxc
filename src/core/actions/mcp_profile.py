"""mcp_profile tool — Telegram profile & message status checker.

Wraps Telethon operations to check:
- Message read status (was the message read?)
- User last online status
- Ignoring detection (read but no response for X time)
- Full profile info (all available fields)

Requires userbot_manager and user (telegram_id) in kwargs.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from telethon import functions
from telethon.errors import FloodWaitError
from telethon.tl.types import (
    Channel,
    Chat,
    User,
    UserStatusOffline,
    UserStatusOnline,
    UserStatusRecently,
    UserStatusLastWeek,
    UserStatusLastMonth,
)

from src.core.actions.tool_registry import tool

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════


def _format_status(status) -> dict[str, Any]:
    """Convert Telethon UserStatus to a readable dict."""
    if status is None:
        return {"status": "hidden", "description": "Статус скрыт"}

    if isinstance(status, UserStatusOnline):
        return {"status": "online", "description": "Сейчас онлайн"}

    if isinstance(status, UserStatusOffline):
        was_online = status.was_online
        if was_online:
            dt = was_online.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            diff = now - dt
            minutes = int(diff.total_seconds() / 60)

            if minutes < 1:
                human = "только что"
            elif minutes < 60:
                human = f"{minutes} мин назад"
            elif minutes < 1440:
                hours = minutes // 60
                human = f"{hours} ч назад"
            else:
                days = minutes // 1440
                human = f"{days} дн назад"

            return {
                "status": "offline",
                "was_online": dt.isoformat(),
                "human_readable": human,
                "description": f"Был(а) в сети {human}",
            }
        return {
            "status": "offline",
            "description": "Был(а) в сети (точное время неизвестно)",
        }

    if isinstance(status, UserStatusRecently):
        return {"status": "recently", "description": "Был(а) в сети недавно"}

    if isinstance(status, UserStatusLastWeek):
        return {
            "status": "last_week",
            "description": "Был(а) в сети на прошлой неделе",
        }

    if isinstance(status, UserStatusLastMonth):
        return {
            "status": "last_month",
            "description": "Был(а) в сети в прошлом месяце",
        }

    return {"status": "unknown", "description": "Статус неизвестен"}


async def _safe_call(coro, max_retries: int = 3):
    """Execute a Telethon coroutine with FloodWait retry handling."""
    for attempt in range(max_retries):
        try:
            return await coro
        except FloodWaitError as e:
            if attempt < max_retries - 1:
                wait_time = min(e.seconds, 30)  # Cap at 30 seconds
                logger.warning(
                    "FloodWait: waiting %d seconds (attempt %d/%d)",
                    wait_time,
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(wait_time)
            else:
                raise


def _get_photo_url(client, entity) -> str | None:
    """Try to get profile photo URL via Telethon."""
    try:
        if entity.photo:
            # Telethon can return a Photo object
            photo = entity.photo
            if hasattr(photo, "photo_id"):
                return f"tg://photo/{entity.id}?id={photo.photo_id}"
    except Exception:
        pass
    return None


def _format_entity(entity, client=None) -> dict[str, Any]:
    """Convert a Telethon entity (User/Chat/Channel) to a comprehensive dict."""
    result: dict[str, Any] = {}

    if isinstance(entity, User):
        result.update(
            {
                "type": "user",
                "user_id": entity.id,
                "first_name": entity.first_name,
                "last_name": entity.last_name,
                "full_name": " ".join(
                    filter(None, [entity.first_name, entity.last_name])
                ),
                "username": entity.username,
                "usernames": [
                    {
                        "username": u.username,
                        "type": getattr(u, "_cls", None) or getattr(u, "type", None),
                    }
                    for u in (entity.usernames or [])
                ],
                "phone": f"+{entity.phone}" if getattr(entity, "phone", None) else None,
                "phone_hidden": not bool(getattr(entity, "phone", None)),
                "lang_code": getattr(entity, "lang_code", None),
                "is_self": bool(entity.is_self),
                "is_contact": bool(entity.contact),
                "is_mutual_contact": bool(entity.mutual_contact),
                "is_bot": bool(entity.bot),
                "is_verified": bool(entity.verified),
                "is_restricted": bool(entity.restricted),
                "is_scam": bool(entity.scam),
                "is_fake": bool(entity.fake),
                "is_premium": bool(entity.premium),
                "is_support": bool(entity.support),
                "is_deleted": bool(entity.deleted),
                "is_close_friend": bool(entity.close_friend),
                "is_min": bool(entity.min),
                "stories_hidden": bool(entity.stories_hidden),
                "contact_require_premium": bool(entity.contact_require_premium),
                "attach_menu_enabled": bool(entity.attach_menu_enabled),
            }
        )

        # Status
        result.update(_format_status(entity.status))

        # Bot-specific fields
        if entity.bot:
            result["bot_info_version"] = getattr(entity, "bot_info_version", None)
            result["bot_chat_history"] = bool(entity.bot_chat_history)
            result["bot_nochats"] = bool(entity.bot_nochats)
            result["bot_inline_geo"] = bool(entity.bot_inline_geo)
            result["bot_can_edit"] = bool(getattr(entity, "can_edit", False))

        # Restriction reason
        if entity.restricted:
            result["restriction_reason"] = getattr(entity, "restriction_reason", None)

        # Emoji status (premium)
        emoji_status = getattr(entity, "emoji_status", None)
        if emoji_status:
            result["emoji_status"] = {
                "document_id": getattr(emoji_status, "document_id", None),
            }

        # Profile color
        profile_color = getattr(entity, "profile_color", None)
        if profile_color is not None:
            result["profile_color"] = profile_color

        # Color
        color = getattr(entity, "color", None)
        if color is not None:
            result["color"] = color

        # Photo
        if entity.photo:
            result["has_photo"] = True
            photo = entity.photo
            result["photo"] = {
                "photo_id": getattr(photo, "photo_id", None),
                "has_video": bool(getattr(photo, "has_video", False)),
                "stripped_thumb": bool(
                    getattr(photo, "stripped_thumb", None) is not None
                ),
            }
        else:
            result["has_photo"] = False

        # Stories
        stories_max_id = getattr(entity, "stories_max_id", None)
        if stories_max_id:
            result["stories_max_id"] = stories_max_id

        # Paid messages
        paid_stars = getattr(entity, "send_paid_messages_stars", None)
        if paid_stars is not None:
            result["send_paid_messages_stars"] = paid_stars

    elif isinstance(entity, (Chat, Channel)):
        result["type"] = "channel" if isinstance(entity, Channel) else "chat"
        result["chat_id"] = entity.id
        result["title"] = entity.title
        result["username"] = getattr(entity, "username", None)
        result["megagroup"] = bool(getattr(entity, "megagroup", False))
        result["gigagroup"] = bool(getattr(entity, "gigagroup", False))
        result["is_broadcast"] = bool(getattr(entity, "broadcast", False))
        result["is_verified"] = bool(getattr(entity, "verified", False))
        result["is_scam"] = bool(getattr(entity, "scam", False))
        result["is_fake"] = bool(getattr(entity, "fake", False))
        result["is_restricted"] = bool(getattr(entity, "restricted", False))
        result["participants_count"] = getattr(entity, "participants_count", None)

    else:
        result["type"] = "unknown"
        result["id"] = getattr(entity, "id", None)

    return result


# ══════════════════════════════════════════════════════════════════════════
# Tool registration
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="tg_profile",
    description=(
        "Проверить статус Telegram-пользователя или сообщения. "
        "Действия:\n"
        "- 'online' — когда пользователь был в сети\n"
        "- 'read' — прочитано ли конкретное сообщение\n"
        "- 'ignore' — игнорирует ли пользователь (прочитал, но не ответил)\n"
        "- 'info' — полная информация о пользователе (имя, био, фото, "
        "статус, бот?, премиум?, контакт?, взаимный контакт?, ограничения, "
        "и т.д.)"
    ),
    category="telegram",
    risk="low",
    params={
        "action": "str — 'online', 'read', 'ignore', 'info'",
        "peer": "str — @username или ID пользователя",
        "message_id": "int|None — ID сообщения (для action='read')",
        "timeout_hours": "float=24 — сколько часов ждать ответа (для action='ignore')",
    },
)
async def tg_profile(
    action: str,
    peer: str = "",
    message_id: int | None = None,
    timeout_hours: float = 24,
    **kwargs: Any,
) -> dict[str, Any]:
    """Check Telegram profile or message status.

    Args:
        action: One of 'online', 'read', 'ignore', 'info'.
        peer: @username or user ID.
        message_id: Message ID (for 'read' action).
        timeout_hours: Hours to consider as ignoring (for 'ignore').

    Returns:
        Dict with status info or error.
    """
    userbot_manager = kwargs.get("userbot_manager")
    _user_val = kwargs.get("user", 0)
    # user may be an int (telegram_id) or a User ORM object — normalise
    if hasattr(_user_val, "telegram_id"):
        telegram_id: int = _user_val.telegram_id
    else:
        telegram_id = int(_user_val)

    if userbot_manager is None:
        return {"error": "userbot_manager not available"}

    client = userbot_manager.get_client(telegram_id)
    if client is None:
        return {"error": "No active Telegram client. Please /login first."}

    # Normalize peer: strip @ prefix if present
    if peer and peer.startswith("@"):
        peer = peer[1:]

    if not peer and action != "ignore":
        return {"error": "peer parameter is required"}

    try:
        if action == "online":
            return await _check_online(client, peer)
        elif action == "read":
            return await _check_read(client, peer, message_id)
        elif action == "ignore":
            return await _check_ignore(client, peer, message_id, timeout_hours)
        elif action == "info":
            return await _check_info(client, peer)
        else:
            return {
                "error": f"Unknown action {action!r}. Valid: online, read, ignore, info"
            }
    except Exception as exc:
        logger.exception("tg_profile(%r, peer=%r) failed", action, peer)
        return {"error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════
# Action handlers
# ══════════════════════════════════════════════════════════════════════════


async def _check_online(client, peer: str) -> dict[str, Any]:
    """Check when user was last online."""
    try:
        entity = await _safe_call(client.get_entity(peer))
    except Exception as e:
        return {"error": f"Не удалось найти пользователя {peer!r}: {e}"}

    if not isinstance(entity, User):
        return {
            "error": f"{peer!r} — это не пользователь (это {type(entity).__name__})"
        }

    status_info = _format_status(entity.status)

    result = {
        "ok": True,
        "peer": peer,
        "user_id": entity.id,
        "first_name": entity.first_name,
        "last_name": entity.last_name,
        "full_name": " ".join(filter(None, [entity.first_name, entity.last_name])),
        **status_info,
    }

    if entity.bot:
        result["is_bot"] = True
        result["description"] = "Это бот — у ботов нет статуса онлайна"

    return result


async def _check_read(client, peer: str, message_id: int | None) -> dict[str, Any]:
    """Check if a specific message was read by the recipient."""
    if message_id is None:
        return {"error": "message_id is required for action='read'"}

    try:
        entity = await _safe_call(client.get_entity(peer))
    except Exception as e:
        return {"error": f"Не удалось найти пользователя {peer!r}: {e}"}

    try:
        # Get the message directly
        msg = await _safe_call(client.get_messages(entity, ids=message_id))
        if msg is None:
            return {"error": f"Сообщение {message_id} не найдено"}

        is_read = False

        if msg.out:
            # For outgoing messages, check if read using dialog's read_outbox_max_id
            # Use get_dialogs with limit=1 to get just this dialog
            async for dialog in client.iter_dialogs(limit=100):
                if dialog.entity and dialog.entity.id == entity.id:
                    if dialog.dialog.read_outbox_max_id >= message_id:
                        is_read = True
                    break

        return {
            "ok": True,
            "peer": peer,
            "message_id": message_id,
            "is_outgoing": bool(msg.out),
            "is_read": is_read,
            "read_status": "Прочитано" if is_read else "Не прочитано",
            "message_date": msg.date.isoformat() if msg.date else None,
            "message_text": (msg.text or "")[:100],
        }
    except Exception as e:
        return {"error": f"Ошибка при проверке сообщения: {e}"}


async def _check_ignore(
    client,
    peer: str,
    message_id: int | None,
    timeout_hours: float,
) -> dict[str, Any]:
    """Check if user is ignoring us: message was read but no response."""
    if message_id is None:
        return {"error": "message_id is required for action='ignore'"}

    try:
        entity = await _safe_call(client.get_entity(peer))
    except Exception as e:
        return {"error": f"Не удалось найти пользователя {peer!r}: {e}"}

    try:
        # Get our message
        msg = await _safe_call(client.get_messages(entity, ids=message_id))
        if msg is None:
            return {"error": f"Сообщение {message_id} не найдено"}

        if not msg.out:
            return {"error": "Сообщение не исходящее — нельзя проверить игнор"}

        # Check if read using dialog's read_outbox_max_id
        is_read = False
        async for dialog in client.iter_dialogs(limit=100):
            if dialog.entity and dialog.entity.id == entity.id:
                if dialog.dialog.read_outbox_max_id >= message_id:
                    is_read = True
                break

        if not is_read:
            return {
                "ok": True,
                "peer": peer,
                "message_id": message_id,
                "is_read": False,
                "is_ignoring": False,
                "description": "Сообщение ещё не прочитано — рано говорить об игноре",
            }

        # Message was read — check if there's a response after it
        # Get recent messages from the chat
        has_response = False
        response_time_hours = None

        async for m in client.iter_messages(entity, limit=10, min_id=message_id):
            # Skip our messages
            if m.out:
                continue
            # This is a message from them after our message
            if m.id > message_id and m.date:
                has_response = True
                msg_date = msg.date.replace(tzinfo=timezone.utc) if msg.date else None
                m_date = m.date.replace(tzinfo=timezone.utc) if m.date else None
                if msg_date and m_date:
                    diff = (m_date - msg_date).total_seconds() / 3600
                    response_time_hours = round(diff, 1)
                break

        now = datetime.now(timezone.utc)
        msg_date = msg.date.replace(tzinfo=timezone.utc) if msg.date else None
        hours_since_sent = (
            round((now - msg_date).total_seconds() / 3600, 1) if msg_date else 0
        )

        if has_response:
            return {
                "ok": True,
                "peer": peer,
                "message_id": message_id,
                "is_read": True,
                "is_ignoring": False,
                "has_response": True,
                "response_time_hours": response_time_hours,
                "description": f"Ответ получен за {response_time_hours} ч — не игнорирует",
            }
        else:
            is_ignoring = hours_since_sent > timeout_hours
            return {
                "ok": True,
                "peer": peer,
                "message_id": message_id,
                "is_read": True,
                "is_ignoring": is_ignoring,
                "has_response": False,
                "hours_since_sent": hours_since_sent,
                "timeout_hours": timeout_hours,
                "description": (
                    f"Прочитано {hours_since_sent} ч назад, ответа нет — "
                    + (
                        "игнорирует 🙄"
                        if is_ignoring
                        else "ещё рано говорить об игноре"
                    )
                ),
            }
    except Exception as e:
        return {"error": f"Ошибка при проверке игнора: {e}"}


async def _check_info(client, peer: str) -> dict[str, Any]:
    """Get FULL profile info — all available fields from Telegram.

    Returns everything Telethon can extract:
    - Basic: name, username, phone, lang, bio
    - Flags: bot, verified, scam, fake, premium, support, deleted, contact, mutual
    - Status: online/offline/recently/last_week/last_month
    - Photo: has_video, photo_id
    - Restrictions, emoji status, profile color
    - For bots: bot_info_version, capabilities
    - Mutual contacts count
    - Common groups count
    """
    try:
        entity = await _safe_call(client.get_entity(peer))
    except Exception as e:
        return {"error": f"Не удалось найти пользователя {peer!r}: {e}"}

    if not isinstance(entity, User):
        return {
            "error": f"{peer!r} — не пользователь (тип: {type(entity).__name__}). "
            f"Используйте tg_profile с action='info' только для пользователей."
        }

    # ── Base fields from entity ──────────────────────────────────────
    result = _format_entity(entity, client)

    # ── Try to get full user info + mutual contacts + common groups ──
    # These are independent API calls — run them in parallel
    async def _get_full():
        try:
            full = await _safe_call(client(functions.users.GetFullUserRequest(entity)))  # type: ignore[arg-type]
            return getattr(full, "full_user", full)
        except Exception:
            logger.debug("Could not fetch full user info for %s", peer)
            return None

    async def _get_mutual():
        try:
            mutual = await _safe_call(client.get_mutual_contacts(entity))
            return len(mutual) if mutual else 0
        except Exception:
            return 0

    async def _get_common():
        try:
            common = await _safe_call(client.get_common_chats(entity))
            if common:
                return [{"id": g.id, "title": g.title} for g in common[:10]]
            return []
        except Exception:
            return []

    fu, mutual_count, common_groups = await asyncio.gather(
        _get_full(), _get_mutual(), _get_common()
    )

    # Apply full user info
    if fu:
        result["bio"] = getattr(fu, "about", None)
        result["common_chats_count"] = getattr(fu, "common_chats_count", 0)
        result["blocked"] = bool(getattr(fu, "blocked", False))
        result["phone_calls_available"] = bool(
            getattr(fu, "phone_calls_available", False)
        )
        result["phone_calls_private"] = bool(getattr(fu, "phone_calls_private", False))
        result["video_calls_available"] = bool(
            getattr(fu, "video_calls_available", False)
        )
        result["voice_messages_forbidden"] = bool(
            getattr(fu, "voice_messages_forbidden", False)
        )
        result["translations_disabled"] = bool(
            getattr(fu, "translations_disabled", False)
        )
        result["stories_pinned_count"] = getattr(fu, "stories_pinned", None)

        # Profile photo from full info
        if getattr(fu, "profile_photo", None):
            result["profile_photo"] = {
                "photo_id": fu.profile_photo.id,
                "has_video": bool(getattr(fu.profile_photo, "has_video", False)),
            }

        # Bot info
        if getattr(fu, "bot_info", None):
            bot_info = fu.bot_info
            result["bot_description"] = getattr(bot_info, "description", None)
            result["bot_commands"] = [
                {"command": c.command, "description": c.description}
                for c in (getattr(bot_info, "commands", None) or [])
            ]

    # Apply mutual contacts
    if mutual_count:
        result["mutual_contacts_count"] = mutual_count

    # Apply common groups
    if common_groups:
        result["common_groups_count"] = len(common_groups)
        result["common_groups"] = common_groups

    # ── Last seen pattern analysis ───────────────────────────────────
    if isinstance(entity.status, UserStatusOffline) and entity.status.was_online:
        dt = entity.status.was_online.replace(tzinfo=timezone.utc)
        result["last_seen_analysis"] = {
            "hour_utc": dt.hour,
            "day_of_week": dt.strftime("%A"),
            "likely_active_hours_utc": f"{dt.hour}:00 ± 2h",
        }

    return result
