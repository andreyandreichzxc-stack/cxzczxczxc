"""Memory Conflict Resolution — автоматическое разрешение противоречивых фактов."""

import logging
from datetime import datetime, timezone
from collections import defaultdict
from src.db.session import get_session
from src.db.repo import get_or_create_user, list_memories, get_contact

logger = logging.getLogger(__name__)


async def find_conflicts(owner_id: int) -> list[dict]:
    """
    Находит противоречивые пары фактов.
    Два факта о том же контакте с противоположным sentiment.
    Возвращает список {contact_name, contact_id, fact1, fact2, date1, date2, confidence1, confidence2}
    """
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        memories = await list_memories(session, owner)
        active = [m for m in memories if m.is_active and m.contact_id and m.sentiment]

        # Группируем по контактам
        by_contact = defaultdict(list)
        for m in active:
            by_contact[m.contact_id].append(m)

        conflicts = []
        for contact_id, facts in by_contact.items():
            if len(facts) < 3:
                continue
            # Ищем пары positive—negative
            positive = [f for f in facts if f.sentiment == "positive" and f.created_at]
            negative = [
                f
                for f in facts
                if f.sentiment in ("negative", "contradictory") and f.created_at
            ]
            if not positive or not negative:
                continue

            # Сравниваем каждый негативный с каждым позитивным
            for n in negative:
                for p in positive:
                    if p.created_at and n.created_at:
                        days_apart = abs((p.created_at - n.created_at).days)
                        if days_apart <= 30:  # конфликт только если <30 дней разницы
                            contact = await get_contact(session, owner, contact_id)
                            conflicts.append(
                                {
                                    "contact_id": contact_id,
                                    "contact_name": contact.display_name
                                    if contact
                                    else str(contact_id),
                                    "fact_positive": p.fact[:100],
                                    "fact_negative": n.fact[:100],
                                    "positive_id": p.id,
                                    "negative_id": n.id,
                                    "date_positive": str(p.created_at.date())
                                    if p.created_at
                                    else "",
                                    "date_negative": str(n.created_at.date())
                                    if n.created_at
                                    else "",
                                    "confidence_positive": p.confidence or 0.5,
                                    "confidence_negative": n.confidence or 0.5,
                                    "days_apart": days_apart,
                                }
                            )
                            if len(conflicts) >= 5:
                                break
                if len(conflicts) >= 5:
                    break
            if len(conflicts) >= 5:
                break
        return conflicts[:5]


async def resolve_conflict(
    owner_id: int, positive_id: int, negative_id: int, resolution: str
) -> bool:
    """
    Разрешает конфликт: деактивирует один из фактов и создаёт resolution-факт.
    resolution: "positive_wins" | "negative_wins" | "both_stale" | "context_explains"
    """
    from src.db.models import Memory
    from src.db.repo import add_memory, link_memories

    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        pos = await session.get(Memory, positive_id)
        neg = await session.get(Memory, negative_id)
        if not pos or not neg:
            return False
        if pos.user_id != owner.id or neg.user_id != owner.id:
            return False

        if resolution == "positive_wins":
            neg.is_active = False
            reason = f"Противоречие разрешено: позитивный факт «{pos.fact[:60]}» актуальнее негативного «{neg.fact[:60]}» ({str(neg.created_at.date()) if neg.created_at else '?'})"
        elif resolution == "negative_wins":
            pos.is_active = False
            reason = f"Противоречие разрешено: негативный факт «{neg.fact[:60]}» актуальнее позитивного «{pos.fact[:60]}» ({str(pos.created_at.date()) if pos.created_at else '?'})"
        elif resolution == "both_stale":
            pos.is_active = False
            neg.is_active = False
            reason = f"Оба факта устарели: «{pos.fact[:50]}» ({str(pos.created_at.date()) if pos.created_at else '?'}) и «{neg.fact[:50]}» ({str(neg.created_at.date()) if neg.created_at else '?'})"
        else:  # context_explains
            reason = f"Контекст объясняет противоречие: «{pos.fact[:50]}» vs «{neg.fact[:50]}»"

        # Сохраняем resolution как факт
        res = await add_memory(
            session,
            owner,
            fact=reason,
            sentiment="neutral",
            source="conflict_resolution",
            contact_id=pos.contact_id,
            importance=0.7,
            memory_tier=2,
        )
        if res:
            await link_memories(
                session,
                owner,
                res.id,
                positive_id,
                weight=0.8,
                relation_type="resolves",
            )
            await link_memories(
                session,
                owner,
                res.id,
                negative_id,
                weight=0.8,
                relation_type="resolves",
            )
        await session.commit()
        return True


def format_conflicts(conflicts: list[dict]) -> str:
    """Форматирует список конфликтов."""
    if not conflicts:
        return "✅ Конфликтов в памяти не обнаружено."
    lines = [f"<b>⚠️ Конфликты в памяти ({len(conflicts)})</b>", ""]
    for i, c in enumerate(conflicts):
        lines.append(
            f"{i + 1}. <b>{c['contact_name']}</b> ({c['days_apart']} дн. разницы)"
        )
        lines.append(
            f"   ✅ {c['fact_positive']} ({c['date_positive']}, conf {c['confidence_positive']:.1f})"
        )
        lines.append(
            f"   ❌ {c['fact_negative']} ({c['date_negative']}, conf {c['confidence_negative']:.1f})"
        )
        lines.append(
            f"   🔍 Контекст: позитив от {c['date_positive']}, негатив от {c['date_negative']}"
        )
        newer = "positive" if c["date_positive"] >= c["date_negative"] else "negative"
        if newer == "positive":
            lines.append(
                f"   💡 Более новый факт — позитивный. Возможно, конфликт разрешён."
            )
        else:
            lines.append(f"   💡 Более новый факт — негативный. Требует внимания.")
        lines.append("")
    return "\n".join(lines)


async def conflict_check_loop(owner_id: int):
    """Фоновый цикл: раз в 12 часов проверяет конфликты."""
    import asyncio
    from src.core.notifier import notifier
    from src.core.timeutil import now_in_tz
    from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

    last_check_date = None
    while True:
        try:
            async with get_session() as session:
                owner = await get_or_create_user(session, owner_id)
                tz_name = owner.settings.timezone if owner.settings else "UTC"
            now = now_in_tz(tz_name)
            today = now.date()
            if now.hour == 12 and last_check_date != today:
                last_check_date = today
                conflicts = await find_conflicts(owner_id)
                if conflicts:
                    text = format_conflicts(conflicts)
                    kb = InlineKeyboardMarkup(
                        inline_keyboard=[
                            [
                                InlineKeyboardButton(
                                    text="✅ Позитив актуален",
                                    callback_data=f"conflict:resolve:{conflicts[0]['positive_id']}:{conflicts[0]['negative_id']}:positive_wins",
                                ),
                            ],
                            [
                                InlineKeyboardButton(
                                    text="❌ Негатив актуален",
                                    callback_data=f"conflict:resolve:{conflicts[0]['positive_id']}:{conflicts[0]['negative_id']}:negative_wins",
                                ),
                            ],
                            [
                                InlineKeyboardButton(
                                    text="⏰ Оба устарели",
                                    callback_data=f"conflict:resolve:{conflicts[0]['positive_id']}:{conflicts[0]['negative_id']}:both_stale",
                                ),
                            ],
                        ]
                    )
                    await notifier.notify(text, reply_markup=kb)
            await asyncio.sleep(600)
        except Exception as e:
            logger.error(f"Conflict check error: {e}")
            await asyncio.sleep(3600)
