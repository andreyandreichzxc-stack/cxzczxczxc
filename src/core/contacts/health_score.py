"""Contact relationship health scoring — 0-100 metric."""

from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


async def get_contact_health(owner_id: int, peer_id: int) -> dict:
    """Returns {score, days_since_last, open_commitments, message_count, reply_ratio, status}"""
    from src.db.session import get_session
    from src.db.repo import get_or_create_user, list_open_commitments

    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id, use_cache=True)

        # 1. Days since last message
        from sqlalchemy import select, func
        from src.db.models import Message

        r = await session.execute(
            select(func.max(Message.date)).where(
                Message.user_id == owner.id, Message.peer_id == peer_id
            )
        )
        last_date = r.scalar_one_or_none()

        if last_date:
            if last_date.tzinfo is None:
                last_date = last_date.replace(tzinfo=timezone.utc)
            days_gap = (datetime.now(timezone.utc) - last_date).days
        else:
            days_gap = 365  # never messaged

        # 2. Open commitments count
        commitments = await list_open_commitments(session, owner, peer_id=peer_id)
        open_count = len(commitments)

        # 3. Message count
        r = await session.execute(
            select(func.count())
            .select_from(Message)
            .where(Message.user_id == owner.id, Message.peer_id == peer_id)
        )
        msg_count = r.scalar_one() or 0

        # 4. Reply ratio (outgoing / total)
        r = await session.execute(
            select(func.count())
            .select_from(Message)
            .where(
                Message.user_id == owner.id,
                Message.peer_id == peer_id,
                Message.is_outgoing.is_(True),
            )
        )
        outgoing = r.scalar_one() or 0

        reply_ratio = outgoing / max(msg_count, 1)

        # --- Scoring ---
        score = 100

        # Days gap penalty: -10 per week of silence (max -60)
        score -= min(days_gap / 7 * 10, 60)

        # Open commitments: minor penalty (-5 each, max -20)
        score -= min(open_count * 5, 20)

        # Low message count: slight penalty (< 10 messages = -10)
        if msg_count < 10:
            score -= 10

        # Reply ratio: boost for balanced (+10 if 0.3-0.7)
        if 0.3 <= reply_ratio <= 0.7:
            score += 10
        elif reply_ratio > 0.9:  # you always reply, they don't → -15
            score -= 15
        elif reply_ratio < 0.1 and msg_count > 5:  # they write, you ignore → -20
            score -= 20

        score = max(0, min(100, round(score)))

        # Status label
        if score >= 80:
            status = "🟢 здоровые"
        elif score >= 50:
            status = "🟡 требуют внимания"
        else:
            status = "🔴 проблемные"

        return {
            "score": score,
            "status": status,
            "days_since_last": days_gap,
            "open_commitments": open_count,
            "message_count": msg_count,
            "reply_ratio": reply_ratio,
        }
