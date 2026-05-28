"""mcp_telegram tool — registered via @tool decorator.

Provides Telegram messaging operations:

- **get_info** — retrieve contact information (name, username, phone,
  bio, last message time, etc.).
- **send** — send a message to a contact (requires user confirmation).

Both actions rely on an active ``TelegramClient`` obtained from the
``UserbotManager`` singleton, which is passed via ``**kwargs``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from telethon import TelegramClient
from telethon.tl.functions.users import GetFullUserRequest
from telethon.tl.types import User as TgUser

from src.core.actions.tool_registry import ToolActionSpec, tool
from src.core.contacts.contact_resolver import resolve
from src.db.repo import get_contact, get_contact_profile, get_or_create_user
from src.db.session import get_session

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════
# Tool: mcp_telegram
# ══════════════════════════════════════════════════════════════════════════


@tool(
    name="mcp_telegram",
    description=(
        "Telegram messaging operations. Supports two actions:\n"
        "- 'get_info' — look up a contact by name/username and return "
        "profile info (display name, username, phone, bio, is_bot, "
        "last message time, archetype, closeness).\n"
        "- 'send' — send a text message to a contact."
    ),
    category="messaging",
    risk="critical",
    requires_confirmation=True,
    actions={
        "get_info": ToolActionSpec(name="get_info", risk="low", read_only=True, idempotent=True),
        "send": ToolActionSpec(
            name="send",
            risk="critical",
            read_only=False,
            destructive=False,
            idempotent=False,
            requires_confirmation=True,
        ),
    },
    params={
        "action": "str — 'get_info' or 'send'",
        "peer": "str — contact name, display name, or @username",
        "text": "str — message text to send (required for action='send')",
        "limit": "int — max fuzzy-search candidates (default 5)",
    },
    output_schema={
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "action": {"type": "string", "description": "get_info or send"},
            "contact": {
                "type": "object",
                "properties": {
                    "display_name": {"type": "string"},
                    "username": {"type": "string"},
                    "bio": {"type": "string"},
                },
                "description": "Contact info (action=get_info)",
            },
            "sent": {
                "type": "boolean",
                "description": "Whether message was sent (action=send)",
            },
            "error": {"type": "string"},
        },
        "required": ["ok"],
    },
)
async def mcp_telegram(
    action: str,
    peer: str = "",
    text: str = "",
    limit: int = 5,
    **kwargs: Any,
) -> dict[str, Any]:
    """Telegram messaging tool.

    Args:
        action: ``"get_info"`` or ``"send"``.
        peer: Contact name, display name, or @username to resolve.
        text: Message text (required when ``action="send"``).
        limit: Maximum number of fuzzy-search candidates to consider.

    Keyword Args:
        userbot_manager: ``UserbotManager`` instance (injected at runtime).
        user: Owner's Telegram ID (int, defaults to 0).

    Returns:
        A dict with the result data or an ``"error"`` key on failure.
    """
    # ── Resolve runtime dependencies ──────────────────────────────────
    userbot_manager = kwargs.get("userbot_manager")
    _user_val = kwargs.get("user", 0)
    # user may be an int (telegram_id) or a User ORM object — normalise
    if hasattr(_user_val, "telegram_id"):
        telegram_id: int = _user_val.telegram_id
    else:
        telegram_id = int(_user_val)

    if userbot_manager is None:
        return {"error": "userbot_manager not available in kwargs"}

    client: TelegramClient | None = userbot_manager.get_client(telegram_id)
    if client is None:
        return {
            "error": ("No active Telegram client for this user. Please /login first.")
        }

    # Parameter validation
    if not peer or not peer.strip():
        return {"error": "peer parameter is required"}

    try:
        if action == "get_info":
            return await _get_info(client, telegram_id, peer.strip(), limit=limit)
        elif action == "send":
            if not bool(kwargs.get("_confirmed", False)):
                return {"error": "requires confirmation"}
            return await _send_message(client, telegram_id, peer.strip(), text)
        else:
            return {
                "error": f"Unknown action {action!r}. Valid actions: get_info, send"
            }
    except Exception as exc:
        logger.exception("mcp_telegram(%r, peer=%r) failed", action, peer)
        return {"error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════
# Action implementations
# ══════════════════════════════════════════════════════════════════════════


async def _get_info(
    client: TelegramClient,
    telegram_id: int,
    peer: str,
    *,
    limit: int = 5,
) -> dict[str, Any]:
    """Look up a contact and return profile information.

    Steps:
    1. Fuzzy-resolve the peer name to a ``peer_id`` via
       ``contact_resolver.resolve``.
    2. Fetch Telethon entity (live data from Telegram servers).
    3. Enrich with local DB data (``Contact`` + ``ContactProfile``).
    """
    # Step 1 — resolve peer name to candidate(s)
    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)

    candidates = await resolve(
        client,
        owner,
        peer,
        limit=limit,
        kinds=("user", "chat", "channel"),
        include_bots=True,
    )

    if not candidates:
        return {"error": f"Contact {peer!r} not found. Try /sync first."}

    target = candidates[0]

    # Step 2 — fetch entity and full user data via Telethon
    entity = await client.get_entity(target.peer_id)

    result: dict[str, Any] = {
        "ok": True,
        "display_name": target.display_name,
        "username": target.username,
        "peer_id": target.peer_id,
        "peer_kind": target.peer_kind,
    }

    # Enrich with Telethon entity fields (for User types)
    if isinstance(entity, TgUser):
        result["is_bot"] = bool(entity.bot)
        result["phone"] = getattr(entity, "phone", None) or None
        result["is_scam"] = bool(getattr(entity, "scam", False))
        result["is_fake"] = bool(getattr(entity, "fake", False))
        result["is_verified"] = bool(getattr(entity, "verified", False))
        result["restricted"] = bool(getattr(entity, "restricted", False))

        # Try to get bio / about from GetFullUserRequest
        try:
            full = await client(GetFullUserRequest(entity.id))
            about = getattr(full, "about", None) or (
                getattr(full.full_user, "about", None)
                if hasattr(full, "full_user")
                else None
            )
            if about:
                result["bio"] = about
        except Exception:
            pass  # bio is optional

    else:
        # For chats / channels — set sensible defaults
        result["is_bot"] = False
        result["phone"] = None
        result["title"] = getattr(entity, "title", None)

    # Step 3 — enrich with local DB data
    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)
        contact = await get_contact(session, owner, target.peer_id)
        if contact is not None:
            result["archetype"] = contact.archetype
            result["is_archived"] = contact.is_archived
            result["folder_names"] = contact.folder_names

        profile = await get_contact_profile(session, owner, target.peer_id)
        if profile is not None:
            result["closeness"] = profile.closeness
            result["closeness_label"] = profile.closeness_label
            result["communication_style"] = profile.communication_style
            result["current_status"] = profile.current_status
            result["relationship_phase"] = profile.relationship_phase

    return result


async def _send_message(
    client: TelegramClient,
    telegram_id: int,
    peer: str,
    text: str,
) -> dict[str, Any]:
    """Send a text message to a resolved contact.

    Resolves *peer* to a Telethon entity, then calls
    ``client.send_message``.  The actual send is executed immediately;
    the ``requires_confirmation=True`` on the ``@tool`` decorator ensures
    the caller (guardrail / LLM orchestrator) asks the user first.
    """
    if not text or not text.strip():
        return {"error": "text parameter is required for action='send'"}

    # Resolve peer
    async with get_session() as session:
        owner = await get_or_create_user(session, telegram_id)

    candidates = await resolve(
        client,
        owner,
        peer,
        limit=5,
        kinds=("user", "chat", "channel"),
        include_bots=True,
    )

    if not candidates:
        return {"error": f"Contact {peer!r} not found. Try /sync first."}

    target = candidates[0]

    # Fetch entity and send
    try:
        entity = await client.get_entity(target.peer_id)
    except Exception as exc:
        logger.exception("Failed to get entity for peer %s", target.peer_id)
        return {"error": f"Cannot resolve peer {peer!r}: {exc}"}

    try:
        sent = await client.send_message(entity, text.strip())
    except Exception as exc:
        logger.exception("Failed to send message to %s", target.peer_id)
        return {"error": f"Send failed: {exc}"}

    return {
        "ok": True,
        "to": target.display_name,
        "username": target.username,
        "text": text.strip(),
        "message_id": sent.id,
        "date": (
            sent.date.isoformat() if isinstance(sent.date, datetime) else str(sent.date)
        ),
    }
