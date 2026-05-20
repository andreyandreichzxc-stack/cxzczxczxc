"""Conflict Predictor — предсказывает потенциальные конфликты на основе паттернов."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone

from sqlalchemy import select

from src.core.notification_queue import notification_queue
from src.db.models import Notification
from src.db.models import Message
from src.db.repo import (
    get_contact,
    get_or_create_user,
    list_active_conversations,
    list_memories,
)
from src.db.session import get_session

logger = logging.getLogger(__name__)


def _naive_utc(dt: datetime | None) -> datetime | None:
    """Приводит datetime к naive UTC для безопасных сравнений с SQLite."""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _last_before(dates: list[datetime], target: datetime) -> datetime | None:
    """Возвращает последний datetime из отсортированного списка перед target."""
    latest = None
    for dt in dates:
        nd = _naive_utc(dt)
        if nd is None:
            continue
        if nd >= target:
            break
        latest = nd
    return latest


async def detect_silence_triggers(owner_id: int) -> list[dict]:
    """
    Анализирует паттерны: контакт получает негативные факты после N часов молчания.

    Returns:
        Список предупреждений:
        [{contact_name, contact_id, silence_hours, current_hours, archetype, negatives}]
    """
    triggers: list[dict] = []
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        memories = await list_memories(session, owner)
        active_convos = await list_active_conversations(session, owner, limit=100)
        conv_by_peer = {c.peer_id: c for c in active_convos}

        # Группируем негативные факты по контактам
        contact_neg: dict[int, list[datetime]] = defaultdict(list)
        for m in memories:
            if m.sentiment == "negative" and m.contact_id and m.created_at:
                nd = _naive_utc(m.created_at)
                if nd is not None:
                    contact_neg[m.contact_id].append(nd)

        for contact_id, neg_dates in contact_neg.items():
            if len(neg_dates) < 2:
                continue
            neg_dates.sort()

            # Находим ConversationState для текущего молчания.
            conv = conv_by_peer.get(contact_id)
            if not conv or not conv.last_outgoing_at:
                continue

            last_out = _naive_utc(conv.last_outgoing_at)
            if last_out is None:
                continue

            outgoing_result = await session.execute(
                select(Message.date)
                .where(
                    Message.user_id == owner.id,
                    Message.peer_id == contact_id,
                    Message.is_outgoing.is_(True),
                    Message.date < neg_dates[-1],
                )
                .order_by(Message.date.asc())
            )
            outgoing_dates = [
                d for d in outgoing_result.scalars().all() if _naive_utc(d) is not None
            ]
            if not outgoing_dates:
                continue

            # Вычисляем молчание перед каждым старым негативом: последнее исходящее до негатива → негатив.
            silence_before_neg: list[float] = []
            for nd in neg_dates[-5:]:
                prior_out = _last_before(outgoing_dates, nd)
                if prior_out is None:
                    continue
                silence = (nd - prior_out).total_seconds() / 3600
                if silence > 0:
                    silence_before_neg.append(silence)

            if not silence_before_neg:
                continue

            avg_silence = sum(silence_before_neg) / len(silence_before_neg)
            if avg_silence < 6:  # меньше 6 часов — не паттерн
                continue

            # Получаем имя и архетип контакта
            contact = await get_contact(session, owner, contact_id)
            name = contact.display_name if contact else str(contact_id)
            archetype = contact.archetype if contact else None

            # Текущее молчание (с момента последнего исходящего)
            now = _naive_utc(datetime.now(timezone.utc))
            assert now is not None
            current_silence = (now - last_out).total_seconds() / 3600

            # Если текущее молчание > 70% от порога — предупредить
            threshold = avg_silence * 0.7
            if current_silence >= threshold:
                triggers.append(
                    {
                        "contact_name": name,
                        "contact_id": contact_id,
                        "silence_hours": round(avg_silence, 1),
                        "current_hours": round(current_silence, 1),
                        "archetype": archetype,
                        "negatives": len(neg_dates),
                    }
                )

    return triggers


def format_conflict_warnings(triggers: list[dict]) -> str:
    """Форматирует предупреждения для отправки владельцу."""
    if not triggers:
        return ""

    lines = ["<b>⚠️ Предупреждение: риск конфликта</b>", ""]
    for t in triggers[:3]:
        name = t["contact_name"]
        hours = t["current_hours"]
        avg = t["silence_hours"]
        neg = t["negatives"]
        arch = t.get("archetype", "")

        emoji_map = {
            "close_friend": "🤝",
            "family": "👨‍👩‍👧",
            "romantic": "💕",
            "colleague": "💼",
            "acquaintance": "👋",
            "toxic": "☠️",
        }
        arch_emoji = emoji_map.get(arch, "")

        lines.append(
            f"{arch_emoji} <b>{name}</b>: молчание {hours:.0f} ч. "
            f"Ранее после {avg:.0f} ч был негатив ({neg} случая). "
            f"Напиши сейчас — предотврати."
        )

    lines.append("")
    lines.append("<i>Используй /send или /chat</i>")
    return "\n".join(lines)


async def conflict_predictor_loop(owner_id: int) -> None:
    """Фоновый цикл: проверка каждые 3 часа."""
    while True:
        try:
            triggers = await detect_silence_triggers(owner_id)
            if triggers:
                text = format_conflict_warnings(triggers)
                await notification_queue.enqueue(
                    topic="conflict",
                    text=text,
                    priority=Notification.PRIORITY_HIGH,
                )
        except Exception as e:
            logger.exception("Conflict predictor error: %s", e)
        await asyncio.sleep(3 * 3600)
