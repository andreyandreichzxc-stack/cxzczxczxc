"""Reply Radar — приоритизированный список диалогов для ответа."""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from src.db.session import get_session
from src.db.repo import (
    get_or_create_user,
    get_contact,
    list_active_conversations,
    fetch_chat_messages,
    list_open_commitments,
    list_memories,
)
from src.db.models import ConversationState
from src.core.memory.temporal_layers import utc_naive, utcnow_naive

logger = logging.getLogger(__name__)


@dataclass
class RadarItem:
    peer_id: int
    contact_name: str
    score: int
    reason: str
    risk_level: str  # low | medium | high
    unread_count: int = 0
    waiting_hours: float = 0
    memory_hints: list = field(default_factory=list)
    latest_snippet: str = ""
    archetype: str | None = None
    reply_window: str = ""  # "вечером 19-21" — из habit_tracker
    suggested_action: str = "reply"  # reply | draft | wait | check


async def collect_reply_radar(owner_id: int, limit: int = 5) -> list[RadarItem]:
    """Собирает приоритизированный список диалогов для ответа."""
    items = []
    now = utcnow_naive()
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        convos = await list_active_conversations(
            session, owner, status="waiting_reply", limit=30
        )
        mems = await list_memories(session, owner)
        commits = await list_open_commitments(session, owner)

        for conv in convos:
            snoozed_until = getattr(conv, "radar_snoozed_until", None)
            if snoozed_until and utc_naive(snoozed_until) > now:
                continue

            # Проверка: владелец ещё не ответил
            if not conv.last_incoming_at:
                continue
            last_incoming = utc_naive(conv.last_incoming_at)
            if conv.last_outgoing_at:
                if utc_naive(conv.last_outgoing_at) >= last_incoming:
                    continue

            contact = await get_contact(session, owner, conv.peer_id)
            name = contact.display_name if contact else str(conv.peer_id)

            # Возраст ожидания (часы)
            waiting_hours = (now - last_incoming).total_seconds() / 3600

            # --- SCORING ---
            score = 0
            reasons = []

            # Возраст ожидания: до 40 баллов
            age_score = min(40, int(waiting_hours / 2))
            score += age_score

            # Unread: до 15
            unread_score = min(15, conv.unread_count * 3)
            score += unread_score

            # Негативные факты / tension
            recent_neg = [
                m
                for m in mems
                if m.contact_id == conv.peer_id
                and m.sentiment == "negative"
                and m.created_at
                and (now - utc_naive(m.created_at)).days < 14
            ]
            if recent_neg:
                score += 20
                reasons.append("недавний негатив")

            # Открытые commitments с этим контактом
            contact_commits = [
                c
                for c in commits
                if c.peer_id == conv.peer_id
                or (hasattr(c, "contact_id") and c.contact_id == conv.peer_id)
            ]
            if contact_commits:
                score += 15
                reasons.append("активные обязательства")

            # Архетип: близкие важнее
            archetype = contact.archetype if contact else None
            if archetype in ("close_friend", "family", "romantic"):
                score += 10
                reasons.append(f"близкий контакт ({archetype})")

            # Sensitivity (из ContactProfile)
            try:
                from src.db.repo import get_contact_profile

                prof = await get_contact_profile(session, owner, conv.peer_id)
                if prof and prof.sensitivity and prof.sensitivity > 0.6:
                    score += 10
                    reasons.append("высокая чувствительность")
            except Exception:
                logger.debug("reply_radar: contact_profile load failed", exc_info=True)
                pass

            # Risk level
            if score >= 60:
                risk = "high"
            elif score >= 35:
                risk = "medium"
            else:
                risk = "low"
            suggested_action = "draft" if risk in {"high", "medium"} else "reply"

            # Memory hints (3 факта)
            hints = [
                m.fact[:80]
                for m in mems
                if m.contact_id == conv.peer_id and m.is_active
            ][:3]

            # Latest snippet
            msgs = await fetch_chat_messages(session, owner, conv.peer_id, limit=3)
            snippet = ""
            for m in msgs:
                if not m.is_outgoing and (m.text or m.transcript):
                    snippet = (m.text or m.transcript or "")[:100]

            # Reply window (из habit_tracker — если есть)
            reply_window = ""
            try:
                from src.core.scheduling.habit_tracker import find_habit_candidates

                contact_mems = [
                    m for m in mems if m.contact_id == conv.peer_id and m.is_active
                ]
                habits = find_habit_candidates(
                    contact_mems, min_occurrences=2, min_weeks=1
                )
                for h in habits:
                    if h["days"] and h["consistency"] >= 0.4:
                        reply_window = f"{h['days']}"
                        break
            except Exception:
                logger.debug("reply_radar: contact_profile load failed", exc_info=True)
                pass

            items.append(
                RadarItem(
                    peer_id=conv.peer_id,
                    contact_name=name,
                    score=score,
                    reason=", ".join(reasons)
                    if reasons
                    else f"ждёт {waiting_hours:.0f}ч",
                    risk_level=risk,
                    unread_count=conv.unread_count,
                    waiting_hours=waiting_hours,
                    memory_hints=hints,
                    latest_snippet=snippet,
                    archetype=archetype,
                    reply_window=reply_window,
                    suggested_action=suggested_action,
                )
            )

    items.sort(key=lambda x: x.score, reverse=True)
    return items[:limit]


def format_radar(items: list[RadarItem]) -> str:
    """Форматирует радар для /today."""
    if not items:
        return "✅ <b>Нет срочных ответов.</b>"
    risk_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
    lines = ["<b>📡 Reply Radar</b>", ""]
    for i, item in enumerate(items):
        emoji = risk_emoji.get(item.risk_level, "⚪")
        window = f" 🕐 {item.reply_window}" if item.reply_window else ""
        lines.append(
            f"{emoji} <b>{item.contact_name}</b> — ждёт {item.waiting_hours:.0f}ч "
            f"({item.unread_count} непроч.) [{item.score}]{window}"
        )
        if item.latest_snippet:
            lines.append(f"   «{item.latest_snippet}»")
        if item.memory_hints:
            lines.append(f"   🧠 {item.memory_hints[0][:70]}")
        lines.append(f"   <i>{item.reason}</i>")
        lines.append("")
    lines.append("<i>/today — полный пульт | /radar — только ответы</i>")
    return "\n".join(lines)
