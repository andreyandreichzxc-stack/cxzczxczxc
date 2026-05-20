"""Weekly digest «Что было на неделе» — comprehensive report.
Собирает метрики со всех подсистем в один сводный отчёт.
Запускается в воскресенье 12:05 (через 5 минут после weekly_summarizer).
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, func

from src.core.notification_queue import notification_queue
from src.core.timeutil import now_in_tz
from src.db.models import (
    Commitment,
    Contact,
    Memory,
    Message,
    Notification,
)
from src.db.repo import get_or_create_user
from src.db.session import get_session

logger = logging.getLogger(__name__)


@dataclass
class WeeklyStats:
    """Собранная статистика за неделю."""

    period_start: datetime
    period_end: datetime

    # Memory
    total_active_facts: int = 0
    new_facts: int = 0
    tier1_facts: int = 0
    tier2_facts: int = 0
    tier3_facts: int = 0
    consolidated_count: int = 0  # фактов, перешедших между tier
    distilled_count: int = 0  # 💡 фактов

    # Commitments
    commitments_created: int = 0
    commitments_fulfilled: int = 0
    commitments_overdue: int = 0
    top_commitments_by_contact: list = field(default_factory=list)  # [(name, count)]

    # Contacts
    messages_total: int = 0
    top_contacts_by_messages: list = field(default_factory=list)  # [(name, count)]
    new_contacts: int = 0
    silent_contacts: list = field(default_factory=list)  # [name]
    top_contacts_by_depth: list = field(default_factory=list)  # [(name, avg_msg_len)]

    # Patterns
    patterns_detected: int = 0
    pattern_changes: list = field(
        default_factory=list
    )  # ["new: periodic_contact Анна"]

    # Habits
    habits_active: int = 0
    habits_improved: list = field(default_factory=list)  # ["спорт: +1"]
    habits_declined: list = field(default_factory=list)  # ["чтение: -2"]

    # Memory health
    fuel_level: float = 0.0
    fragmentation: float = 0.0
    health_warnings: list = field(default_factory=list)

    # Emotional trend
    avg_sentiment: float = 0.0  # -1..+1
    sentiment_trend: str = ""  # "improving", "declining", "stable"

    # Distilled knowledge
    top_distilled: list = field(default_factory=list)  # [fact_text]


class WeeklyDigestBuilder:
    """Собирает данные и формирует отчёт."""

    async def gather_stats(self, session, owner) -> WeeklyStats:
        """Собрать все метрики за последние 7 дней.

        owner: объект User (содержит id — DB primary key, и telegram_id).
        """
        db_user_id = owner.id
        telegram_id = owner.telegram_id

        now = datetime.now(timezone.utc)
        week_ago = now - timedelta(days=7)

        stats = WeeklyStats(
            period_start=week_ago,
            period_end=now,
        )

        # === Memory stats ===
        # Total active
        result = await session.execute(
            select(func.count())
            .select_from(Memory)
            .where(Memory.is_active == True, Memory.user_id == db_user_id)
        )
        stats.total_active_facts = result.scalar() or 0

        # By tier
        result = await session.execute(
            select(func.count())
            .select_from(Memory)
            .where(
                Memory.is_active == True,
                Memory.user_id == db_user_id,
                Memory.memory_tier == 1,
            )
        )
        stats.tier1_facts = result.scalar() or 0

        result = await session.execute(
            select(func.count())
            .select_from(Memory)
            .where(
                Memory.is_active == True,
                Memory.user_id == db_user_id,
                Memory.memory_tier == 2,
            )
        )
        stats.tier2_facts = result.scalar() or 0

        result = await session.execute(
            select(func.count())
            .select_from(Memory)
            .where(
                Memory.is_active == True,
                Memory.user_id == db_user_id,
                Memory.memory_tier == 3,
            )
        )
        stats.tier3_facts = result.scalar() or 0

        # New facts this week
        result = await session.execute(
            select(func.count())
            .select_from(Memory)
            .where(
                Memory.is_active == True,
                Memory.user_id == db_user_id,
                Memory.created_at >= week_ago,
            )
        )
        stats.new_facts = result.scalar() or 0

        # Consolidated (tagged with 'consolidated')
        result = await session.execute(
            select(func.count())
            .select_from(Memory)
            .where(
                Memory.is_active == True,
                Memory.user_id == db_user_id,
                Memory.tags.ilike("%consolidated%"),
            )
        )
        stats.consolidated_count = result.scalar() or 0

        # Distilled
        result = await session.execute(
            select(func.count())
            .select_from(Memory)
            .where(
                Memory.is_active == True,
                Memory.user_id == db_user_id,
                Memory.tags.ilike("%💡%"),
            )
        )
        stats.distilled_count = result.scalar() or 0

        # === Commitment stats ===
        result = await session.execute(
            select(func.count())
            .select_from(Commitment)
            .where(
                Commitment.user_id == db_user_id,
                Commitment.created_at >= week_ago,
            )
        )
        stats.commitments_created = result.scalar() or 0

        result = await session.execute(
            select(func.count())
            .select_from(Commitment)
            .where(
                Commitment.user_id == db_user_id,
                Commitment.status == "done",
                Commitment.created_at >= week_ago,
            )
        )
        stats.commitments_fulfilled = result.scalar() or 0

        # Overdue: status="open" and deadline < now
        result = await session.execute(
            select(func.count())
            .select_from(Commitment)
            .where(
                Commitment.user_id == db_user_id,
                Commitment.status == "open",
                Commitment.deadline_at < now,
            )
        )
        stats.commitments_overdue = result.scalar() or 0

        # === Contact stats ===
        # Messages this week
        result = await session.execute(
            select(func.count())
            .select_from(Message)
            .where(Message.user_id == db_user_id, Message.date >= week_ago)
        )
        stats.messages_total = result.scalar() or 0

        # Top contacts (by messages)
        top_msg_query = (
            select(
                Contact.id,
                Contact.display_name,
                func.count(Message.id).label("cnt"),
            )
            .join(Message, Message.peer_id == Contact.peer_id)
            .where(
                Message.user_id == db_user_id,
                Contact.user_id == db_user_id,
                Message.date >= week_ago,
            )
            .group_by(Contact.id)
            .order_by(func.count(Message.id).desc())
            .limit(5)
        )
        top_msg_result = await session.execute(top_msg_query)
        stats.top_contacts_by_messages = [
            (r.display_name or f"id:{r.id}", r.cnt) for r in top_msg_result.all()
        ]

        # New contacts: contacts whose first message appeared this week
        new_contacts_subq = (
            select(func.min(Message.date).label("first_msg"))
            .where(Message.user_id == db_user_id)
            .group_by(Message.peer_id)
            .having(func.min(Message.date) >= week_ago)
            .subquery()
        )
        result = await session.execute(
            select(func.count()).select_from(new_contacts_subq)
        )
        stats.new_contacts = result.scalar() or 0

        # === Silent contacts (no messages >5 days) ===
        silent_cutoff = now - timedelta(days=5)
        silent_query = (
            select(Contact.display_name, func.max(Message.date).label("last_msg"))
            .join(Message, Message.peer_id == Contact.peer_id)
            .where(
                Contact.user_id == db_user_id,
                Contact.peer_kind == "user",
                Contact.is_bot == False,
            )
            .group_by(Contact.id)
            .having(func.max(Message.date) < silent_cutoff)
            .limit(5)
        )
        silent_result = await session.execute(silent_query)
        stats.silent_contacts = [r.display_name for r in silent_result.all()]

        # === Memory health (from memory_fuel) ===
        try:
            from src.core.memory_fuel import get_fuel_stats

            fuel_data = await get_fuel_stats(telegram_id)
            total = max(fuel_data.get("total_contacts", 1), 1)
            fueled = fuel_data.get("fueled", 0)
            critical = fuel_data.get("critical", 0)
            stats.fuel_level = (fueled - critical) / total
            stats.fuel_level = max(0.0, min(1.0, stats.fuel_level))
        except Exception:
            stats.fuel_level = 0.85  # заглушка

        # === Habits (from habit_tracker) ===
        try:
            from src.db.repo import list_memories as _list_memories
            from src.core.habit_tracker import find_habit_candidates

            memories = await _list_memories(session, owner)
            habits = find_habit_candidates(memories, min_occurrences=3, min_weeks=1)
            stats.habits_active = len(habits)
            for h in habits[:5]:
                topic = list(h.get("keywords", set()))[:2]
                topic_str = ", ".join(topic) if topic else "привычка"
                stats.habits_improved.append(f"{topic_str}: +{h.get('count', 0)}")
        except Exception:
            pass

        # === Distilled knowledge ===
        distilled_result = await session.execute(
            select(Memory)
            .where(
                Memory.is_active == True,
                Memory.user_id == db_user_id,
                Memory.tags.ilike("%💡%"),
            )
            .order_by(Memory.use_count.desc())
            .limit(5)
        )
        stats.top_distilled = [m.fact for m in distilled_result.scalars().all()]

        return stats

    def format_report(self, stats: WeeklyStats) -> str:
        """
        Форматирует WeeklyStats в читаемый отчёт.
        """
        period_str = (
            f"{stats.period_start.strftime('%d.%m')} – "
            f"{stats.period_end.strftime('%d.%m')}"
        )

        lines = [
            f"📊 **Что было на неделе** ({period_str})",
            "━" * 28,
            "",
        ]

        # Memory
        lines.append(
            f"🧠 **Память**: +{stats.new_facts} фактов, "
            f"{stats.consolidated_count} сжато в tier-2, "
            f"{stats.distilled_count} 💡 дистиллировано"
        )
        lines.append(
            f"   Всего активно: {stats.total_active_facts} "
            f"(t1:{stats.tier1_facts} t2:{stats.tier2_facts} t3:{stats.tier3_facts})"
        )

        # Commitments
        if (
            stats.commitments_created
            or stats.commitments_fulfilled
            or stats.commitments_overdue
        ):
            lines.append(
                f"📝 **Обещания**: +{stats.commitments_created} создано, "
                f"{stats.commitments_fulfilled} выполнено, "
                f"{stats.commitments_overdue} просрочено"
            )

        # Contacts
        if stats.messages_total:
            lines.append(f"💬 **Сообщений**: {stats.messages_total}")

        if stats.top_contacts_by_messages:
            top = ", ".join(
                f"{name} ({cnt})" for name, cnt in stats.top_contacts_by_messages[:3]
            )
            lines.append(f"👥 **Топ контактов**: {top}")

        if stats.new_contacts:
            lines.append(f"🆕 **Новых контактов**: {stats.new_contacts}")

        # Silent contacts
        if stats.silent_contacts:
            silent = ", ".join(stats.silent_contacts[:5])
            lines.append(f"⚠️  **Тишина**: {silent} — >5 дней без ответа")

        # Patterns
        if stats.patterns_detected:
            lines.append(f"📈 **Паттерны**: обнаружено {stats.patterns_detected}")

        # Habits
        if stats.habits_improved or stats.habits_declined:
            habit_lines = []
            for h in stats.habits_improved:
                habit_lines.append(f"   ✅ {h}")
            for h in stats.habits_declined:
                habit_lines.append(f"   ❌ {h}")
            if habit_lines:
                lines.append("🏋️ **Привычки**:")
                lines.extend(habit_lines)

        # Memory health
        frag_label = (
            "низкая"
            if stats.fragmentation < 0.3
            else "средняя"
            if stats.fragmentation < 0.6
            else "высокая"
        )
        lines.append(
            f"❤️  **Здоровье памяти**: fuel {stats.fuel_level:.0%}, "
            f"фрагментация {frag_label}"
        )

        # Distilled knowledge
        if stats.top_distilled:
            lines.append("")
            lines.append("💡 **Знания недели**:")
            for fact in stats.top_distilled[:5]:
                lines.append(f"• {fact[:120]}")

        return "\n".join(lines)


async def weekly_digest_loop(owner_id: int) -> None:
    """
    Запуск в воскресенье 12:05 (через 5 минут после weekly_summarizer).

    owner_id: telegram_id владельца.
    """
    last_run_date = None

    while True:
        try:
            async with get_session() as session:
                owner = await get_or_create_user(session, owner_id)
                tz_name = (
                    owner.settings.timezone
                    if owner.settings and owner.settings.timezone
                    else "UTC"
                )

            now = now_in_tz(tz_name)
            # Воскресенье, между 12:00 и 13:00
            if (
                now.weekday() == 6
                and 12 <= now.hour < 13
                and last_run_date != now.date()
            ):
                last_run_date = now.date()

                builder = WeeklyDigestBuilder()
                async with get_session() as session:
                    # Передаём объект User (содержит и id, и telegram_id)
                    owner_db = await get_or_create_user(session, owner_id)
                    stats = await builder.gather_stats(session, owner_db)
                    report = builder.format_report(stats)

                await notification_queue.enqueue(
                    topic="weekly_digest",
                    text=report,
                    priority=Notification.PRIORITY_MEDIUM,
                    category="weekly_report",
                )
                logger.info("Weekly digest sent for owner %d", owner_id)

            await asyncio.sleep(3600)  # проверка раз в час
        except Exception as e:
            logger.error("Weekly digest error: %s", e)
            await asyncio.sleep(3600)


# Синглтон
weekly_digest_builder = WeeklyDigestBuilder()
