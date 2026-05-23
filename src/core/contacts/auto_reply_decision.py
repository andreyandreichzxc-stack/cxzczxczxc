"""Unified auto-reply decision layer.

Consolidates cooldown, spam, archive, group, bot, offline-only,
and recent-owner-message checks into a single async decision function.
Auto-reply handlers call ``decide()`` instead of duplicating inline logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import AutoReplyLog, Contact, Message, User

logger = logging.getLogger(__name__)


class AutoReplyVerdict(Enum):
    """Outcome of the auto-reply decision."""

    SEND = "send"
    SKIP_COOLDOWN = "cooldown"
    SKIP_BOT = "bot"
    SKIP_GROUP = "group"
    SKIP_ARCHIVED = "archived"
    SKIP_SPAM = "spam"
    SKIP_RECENT_MY_MESSAGE = "recent"
    SKIP_OFFLINE_ONLY = "offline"


@dataclass
class AutoReplyChoice:
    """Result of calling ``decide()``."""

    verdict: AutoReplyVerdict
    style: str = "default"  # "default" | "contact" | "global"
    reason: str = ""


async def decide(
    session: AsyncSession,
    owner: User,
    peer_id: int,
    *,
    is_private: bool,
    is_bot: bool = False,
    contact: Contact | None = None,
    is_online: bool = False,
    msg_text: str = "",
) -> AutoReplyChoice:
    """Central auto-reply decision function.

    Checks are applied in the following order (first rejection wins):

      1. **Bot** → ``SKIP_BOT``
      2. **Group** (not a PM) → ``SKIP_GROUP``
      3. **Archived** contact + ``ignore_archived`` → ``SKIP_ARCHIVED``
      4. **Cooldown** (last reply younger than ``auto_reply_cooldown_min``)
         → ``SKIP_COOLDOWN``
      5. **Spam** (5+ incoming messages from *peer_id* in the last 60 seconds)
         → ``SKIP_SPAM``
      6. **Recent owner message** (owner's last outgoing younger than 2 minutes)
         → ``SKIP_RECENT_MY_MESSAGE``
      7. **Offline-only mode** + owner is currently online → ``SKIP_OFFLINE_ONLY``
      8. **Style selection** — contact profile > global profile > ``"default"``
      9. All checks pass → ``SEND``

    Parameters
    ----------
    session:
        Open database session (the caller is responsible for committing).
    owner:
        The ``User`` whose settings are evaluated.
    peer_id:
        Telegram peer-id of the message sender.
    is_private:
        ``True`` when the incoming message is a private chat.
    is_bot:
        ``True`` when the sender is a Telegram bot.
    contact:
        The ``Contact`` ORM object for *peer_id*, if already loaded.
        When ``None`` the archive check is skipped (no known contact).
    is_online:
        ``True`` when the owner is currently considered online.
        The caller is responsible for determining this (e.g. via
        ``_check_and_track_offline``).
    msg_text:
        Raw text of the incoming message (used only for logging at the moment).

    Returns
    -------
    AutoReplyChoice
        Verdict, chosen *style* label, and a human-readable *reason*.
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    # ── 1. Bot sender ──────────────────────────────────────────────────────
    if is_bot:
        return AutoReplyChoice(
            verdict=AutoReplyVerdict.SKIP_BOT,
            reason="Sender is a bot",
        )

    # ── 2. Group (not private message) ─────────────────────────────────────
    if not is_private:
        return AutoReplyChoice(
            verdict=AutoReplyVerdict.SKIP_GROUP,
            reason="Message is from a group chat, not a private chat",
        )

    # ── 3. Archived contact ────────────────────────────────────────────────
    if (
        owner.settings
        and owner.settings.ignore_archived
        and contact is not None
        and contact.is_archived
    ):
        return AutoReplyChoice(
            verdict=AutoReplyVerdict.SKIP_ARCHIVED,
            reason="Contact is archived and ignore_archived is enabled",
        )

    # ── 4. Cooldown — last reply younger than N minutes ────────────────────
    cooldown = owner.settings.auto_reply_cooldown_min if owner.settings else 30
    threshold_cooldown = now - timedelta(minutes=cooldown)
    result = await session.execute(
        select(AutoReplyLog)
        .where(
            AutoReplyLog.user_id == owner.id,
            AutoReplyLog.peer_id == peer_id,
            AutoReplyLog.created_at >= threshold_cooldown,
        )
        .limit(1)
    )
    if result.scalar_one_or_none() is not None:
        return AutoReplyChoice(
            verdict=AutoReplyVerdict.SKIP_COOLDOWN,
            reason=f"Already replied within {cooldown} min cooldown",
        )

    # ── 5. Spam — 5+ incoming messages in the last 60 seconds ─────────────
    spam_threshold = now - timedelta(seconds=60)
    result = await session.execute(
        select(func.count(Message.id)).where(
            Message.user_id == owner.id,
            Message.peer_id == peer_id,
            Message.is_outgoing == False,
            Message.date >= spam_threshold,
        )
    )
    msg_count = result.scalar() or 0
    if msg_count >= 5:
        return AutoReplyChoice(
            verdict=AutoReplyVerdict.SKIP_SPAM,
            reason=f"{msg_count} incoming messages from peer in last 60s",
        )

    # ── 6. Recent owner message (owner wrote < 2 min ago) ─────────────────
    recent_self_threshold = now - timedelta(minutes=2)
    result = await session.execute(
        select(Message.date)
        .where(
            Message.user_id == owner.id,
            Message.peer_id == peer_id,
            Message.is_outgoing == True,
            Message.date >= recent_self_threshold,
        )
        .order_by(Message.date.desc())
        .limit(1)
    )
    last_my_msg = result.scalar_one_or_none()
    if last_my_msg is not None:
        return AutoReplyChoice(
            verdict=AutoReplyVerdict.SKIP_RECENT_MY_MESSAGE,
            reason="Owner sent a message within the last 2 minutes",
        )

    # ── 7. Offline-only mode + owner is online ────────────────────────────
    mode = owner.settings.auto_mode if owner.settings else "offline_only"
    if mode == "offline_only" and is_online:
        return AutoReplyChoice(
            verdict=AutoReplyVerdict.SKIP_OFFLINE_ONLY,
            reason="Owner is online and auto_mode is offline_only",
        )

    # ── 8. Style selection ─────────────────────────────────────────────────
    style = _select_style(contact, owner)

    return AutoReplyChoice(
        verdict=AutoReplyVerdict.SEND,
        style=style,
        reason="All checks passed",
    )


# ── internal helpers ────────────────────────────────────────────────────────


def _select_style(contact: Contact | None, owner: User) -> str:
    """Pick reply style label from available profiles.

    Priority: contact-specific profile > owner global profile > ``"default"``.
    """
    if contact and contact.style_profile:
        return "contact"
    if owner.global_style_profile:
        return "global"
    return "default"
