"""Inbox priority — rank incoming messages by urgency."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.core.infra.text_sanitizer import sanitize_html
from sqlalchemy import func, select, desc

logger = logging.getLogger(__name__)


async def rank_inbox(owner_telegram_id: int, limit: int = 10) -> list[dict]:
    """Rank incoming (unread/unreplied) conversations by priority.

    Returns list of dicts sorted by priority_score (highest first):
        {peer_id, peer_name, priority_score, urgency, reasons, last_message, hours_unreplied, health_score}
    """
    from src.db.session import get_session
    from src.db.repo import (
        get_or_create_user,
        list_active_conversations,
        list_open_commitments,
    )
    from src.db.models import Contact, Message

    async with get_session() as session:
        owner = await get_or_create_user(session, owner_telegram_id)
        now = datetime.now(timezone.utc)

        # Get active conversations where user hasn't replied
        convs = await list_active_conversations(
            session, owner, status="waiting_reply", limit=limit * 2
        )

        # Get open commitments for urgency boost
        commitments = await list_open_commitments(session, owner)
        commitment_peer_ids = {c.peer_id for c in commitments}

        # ── Batch load contacts ────────────────────────────────────
        peer_ids = [c.peer_id for c in convs]
        if peer_ids:
            contacts_r = await session.execute(
                select(Contact).where(
                    Contact.peer_id.in_(peer_ids),
                    Contact.user_id == owner.id,
                )
            )
            contacts_by_peer = {c.peer_id: c for c in contacts_r.scalars().all()}
        else:
            contacts_by_peer = {}

        # ── Batch load health data ─────────────────────────────────
        if peer_ids:
            # Max message date per peer
            max_date_r = await session.execute(
                select(Message.peer_id, func.max(Message.date).label("max_date"))
                .where(
                    Message.user_id == owner.id,
                    Message.peer_id.in_(peer_ids),
                )
                .group_by(Message.peer_id)
            )
            max_dates = {r[0]: r[1] for r in max_date_r.all()}

            # Total message count per peer
            count_r = await session.execute(
                select(Message.peer_id, func.count().label("cnt"))
                .where(
                    Message.user_id == owner.id,
                    Message.peer_id.in_(peer_ids),
                )
                .group_by(Message.peer_id)
            )
            msg_counts = {r[0]: r[1] for r in count_r.all()}

            # Outgoing message count per peer
            out_r = await session.execute(
                select(Message.peer_id, func.count().label("cnt"))
                .where(
                    Message.user_id == owner.id,
                    Message.peer_id.in_(peer_ids),
                    Message.is_outgoing.is_(True),
                )
                .group_by(Message.peer_id)
            )
            outgoing_counts = {r[0]: r[1] for r in out_r.all()}
        else:
            max_dates = {}
            msg_counts = {}
            outgoing_counts = {}

        ranked = []

        for c in convs:
            # Skip if user already replied
            if c.last_outgoing_at and c.last_incoming_at:
                last_out = (
                    c.last_outgoing_at.replace(tzinfo=timezone.utc)
                    if c.last_outgoing_at.tzinfo is None
                    else c.last_outgoing_at
                )
                last_in = (
                    c.last_incoming_at.replace(tzinfo=timezone.utc)
                    if c.last_incoming_at.tzinfo is None
                    else c.last_incoming_at
                )
                if last_out > last_in:
                    continue

            # Calculate hours since last incoming
            if not c.last_incoming_at:
                continue

            last_in = (
                c.last_incoming_at.replace(tzinfo=timezone.utc)
                if c.last_incoming_at.tzinfo is None
                else c.last_incoming_at
            )
            hours_unreplied = (now - last_in).total_seconds() / 3600

            if hours_unreplied < 1:
                continue  # too recent, not yet "waiting"

            # Get last message text
            last_msg = await session.execute(
                select(Message.text)
                .where(
                    Message.user_id == owner.id,
                    Message.peer_id == c.peer_id,
                    Message.is_outgoing.is_(False),
                )
                .order_by(desc(Message.date))
                .limit(1)
            )
            last_text = last_msg.scalar() or ""

            # Priority scoring
            priority = 0.0
            reasons = []

            # 1. Time urgency: +0.5 per day, max 3.0
            days = hours_unreplied / 24
            time_score = min(days * 0.5, 3.0)
            priority += time_score
            if days > 1:
                reasons.append(f"{int(days)}дн без ответа")

            # 2. Commitment boost: +2.0 if there's an open commitment
            if c.peer_id in commitment_peer_ids:
                priority += 2.0
                reasons.append("открытое обязательство")

            # 3. Contact health: lower health = higher priority (computed from batched data)
            health_score = 100
            try:
                peer_id = c.peer_id
                max_date = max_dates.get(peer_id)
                msg_count = msg_counts.get(peer_id, 0)
                outgoing = outgoing_counts.get(peer_id, 0)

                if max_date:
                    if max_date.tzinfo is None:
                        max_date = max_date.replace(tzinfo=timezone.utc)
                    days_gap = (datetime.now(timezone.utc) - max_date).days
                else:
                    days_gap = 365

                open_count = 1 if peer_id in commitment_peer_ids else 0
                reply_ratio = outgoing / max(msg_count, 1)

                score_val = 100.0
                score_val -= min(days_gap / 7 * 10, 60)
                score_val -= min(open_count * 5, 20)
                if msg_count < 10:
                    score_val -= 10
                if 0.3 <= reply_ratio <= 0.7:
                    score_val += 10
                elif reply_ratio > 0.9:
                    score_val -= 15
                elif reply_ratio < 0.1 and msg_count > 5:
                    score_val -= 20

                health_score = max(0, min(100, round(score_val)))

                if health_score < 50:
                    priority += 2.0
                    reasons.append(f"здоровье {health_score}")
                elif health_score < 70:
                    priority += 1.0
                    reasons.append("требует внимания")
            except Exception:
                logger.debug(
                    "inbox_priority: health_score failed for peer %s",
                    c.peer_id,
                    exc_info=True,
                )

            # 4. Urgency keywords in last message
            urgency_words = {
                "срочно",
                "asap",
                "сейчас",
                "быстрее",
                "важно",
                "дедлайн",
                "горит",
                "пж",
                "пожалуйста",
            }
            if last_text:
                text_lower = last_text.lower()
                if any(w in text_lower for w in urgency_words):
                    priority += 1.5
                    reasons.append("срочное сообщение")

            # Get contact name (from batch-loaded dict)
            contact = contacts_by_peer.get(c.peer_id)
            peer_name = contact.display_name if contact else str(c.peer_id)

            # Determine urgency label
            if priority >= 4.0:
                urgency = "🔴 critical"
            elif priority >= 2.5:
                urgency = "🟡 high"
            elif priority >= 1.0:
                urgency = "🟢 normal"
            else:
                urgency = "⚪ low"

            ranked.append(
                {
                    "peer_id": c.peer_id,
                    "peer_name": peer_name,
                    "priority_score": round(priority, 1),
                    "urgency": urgency,
                    "reasons": reasons,
                    "last_message": last_text[:100] if last_text else "",
                    "hours_unreplied": round(hours_unreplied, 1),
                    "health_score": health_score,
                }
            )

        # Sort by priority descending
        ranked.sort(key=lambda x: x["priority_score"], reverse=True)

        return ranked[:limit]


async def format_inbox(ranked: list[dict]) -> str:
    """Format ranked inbox into a readable HTML message for Telegram."""
    if not ranked:
        return "📥 <b>Входящие пусты</b> — не на что отвечать!"

    lines = ["📥 <b>Приоритетные входящие</b>", ""]

    for i, item in enumerate(ranked[:10], 1):
        urgency_icon = item["urgency"].split()[0]  # emoji
        reasons_str = ", ".join(item["reasons"]) if item["reasons"] else ""
        preview = item["last_message"][:60].replace("\n", " ")

        lines.append(
            f"{urgency_icon} {i}. <b>{sanitize_html(item['peer_name'])}</b> "
            f"({item['hours_unreplied']:.0f}ч) [{item['priority_score']}]"
        )
        if reasons_str:
            lines.append(f"   <i>{sanitize_html(reasons_str)}</i>")
        if preview:
            lines.append(f"   «{sanitize_html(preview)}»")
        lines.append("")

    return "\n".join(lines)
