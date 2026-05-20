"""Memory Fuel Gauge — отслеживает «истощение» памяти по контактам."""

import logging
from datetime import datetime, timedelta, timezone

from src.db.repo import get_or_create_user, list_contacts, list_memories
from src.db.session import get_session

logger = logging.getLogger(__name__)


async def get_fuel_stats(owner_id: int) -> dict:
    """
    Возвращает статистику «топлива» памяти (кэшируется на 5 минут).

    Возвращает словарь:
        total_contacts  — общее количество контактов (user, не боты)
        fueled          — контакты с активной памятью
        depleted        — истощённые (но не критично)
        critical        — критически истощённые
        depleted_contacts — список истощённых контактов с деталями
    """
    from src.core.actions.stats_cache import get_cached, set_cache

    cache_key = f"fuel:{owner_id}"
    cached = await get_cached(cache_key)
    if cached is not None:
        return cached

    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        contacts = await list_contacts(
            session, owner, kinds=("user",), include_bots=False
        )
        memories = await list_memories(session, owner)
        now = datetime.now(timezone.utc)

        # Группировка памяти по contact_id (peer_id)
        per_contact: dict[int, list] = {}
        for m in memories:
            if m.is_active and m.contact_id is not None:
                per_contact.setdefault(m.contact_id, []).append(m)

        depleted_contacts: list[dict] = []
        fueled_count = 0
        critical_count = 0

        for contact in contacts:
            facts = per_contact.get(contact.peer_id, [])
            if not facts:
                # контакт без фактов — не истощён, просто нет данных
                continue

            # Последняя дата среди фактов
            dates = [f.created_at for f in facts if f.created_at]
            last_fact = max(dates) if dates else now
            avg_conf = sum(f.confidence for f in facts) / len(facts)
            days_since = (now - last_fact).days

            # Истощён: >14 дней без новых фактов ИЛИ средний confidence < 0.3
            if days_since > 14 or avg_conf < 0.3:
                name = contact.display_name or str(contact.peer_id)
                if days_since > 21 or avg_conf < 0.15:
                    critical_count += 1
                depleted_contacts.append(
                    {
                        "peer_id": contact.peer_id,
                        "name": name,
                        "days_since": days_since,
                        "avg_confidence": round(avg_conf, 2),
                        "fact_count": len(facts),
                        "critical": days_since > 21 or avg_conf < 0.15,
                    }
                )
            else:
                fueled_count += 1

    result = {
        "total_contacts": len(contacts),
        "fueled": fueled_count,
        "depleted": len(depleted_contacts),
        "critical": critical_count,
        "depleted_contacts": depleted_contacts,
    }
    await set_cache(cache_key, result)
    return result


def format_fuel_line(stats: dict) -> str:
    """Однострочный индикатор для вставки в брифинг / треды."""
    total = stats["total_contacts"]
    fueled = stats["fueled"]
    depleted = stats["depleted"]
    critical = stats["critical"]
    if depleted == 0 and critical == 0:
        return f"🟢 Память активна ({fueled}/{total})"
    if critical > 0:
        return f"🔴 Память: {critical} критично, {depleted} истощено, {fueled} активно"
    return f"🟡 Память: {depleted} истощено, {fueled} активно"


def format_depleted_contacts(stats: dict) -> str:
    """Список истощённых контактов с предложением действий."""
    if not stats["depleted_contacts"]:
        return ""
    lines = ["", "<b>🪫 Истощённые контакты:</b>"]
    for dc in stats["depleted_contacts"][:8]:
        icon = "🔴" if dc["critical"] else "🟡"
        lines.append(
            f"{icon} <b>{dc['name']}</b> — {dc['days_since']} дн., "
            f"confidence {dc['avg_confidence']}, {dc['fact_count']} фактов"
        )
    lines.append("💡 <i>/chat Контакт — обновить память</i>")
    return "\n".join(lines)
