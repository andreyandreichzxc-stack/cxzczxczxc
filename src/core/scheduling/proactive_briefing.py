import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import partial

from sqlalchemy import case, select, func, and_, or_, desc

from src.config import settings
from src.core.memory.memory_fuel import get_fuel_stats
from src.core.memory.memory_health import (
    calculate_health_score,
    compute_emotional_trend,
    format_health,
    format_health_compact,
)
from src.core.scheduling.notification_queue import notification_queue
from src.core.contacts.reply_radar import collect_reply_radar
from src.core.memory.temporal_layers import format_layer_stats, get_layer_stats
from src.db.models import (
    Commitment,
    Contact,
    ConversationState,
    Memory,
    Message,
    Notification,
)
from src.db.repo import (
    get_memory_stats,
    get_or_create_user,
    list_active_conversations,
    list_memories,
    list_open_commitments,
)
from src.db.session import get_session
from src.core.infra.task_manager import task_manager

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Backward compat: старые структуры для ручной команды /briefing
# (используется в digest_cmd.py)
# ═══════════════════════════════════════════════════════════════════


@dataclass
class BriefingData:
    urgent_count: int = 0
    unread_total: int = 0
    waiting_reply: list[dict] = field(default_factory=list)
    overdue_commitments: list[dict] = field(default_factory=list)
    today_commitments: list[dict] = field(default_factory=list)
    recent_memories: int = 0
    memory_stats: dict = None  # статистика по памяти
    fuel_stats: dict = None  # статистика топлива памяти
    bridges: list = None  # смысловые мосты между контактами
    cross_insights: list = None  # инсайты по связям между контактами
    tag_stats: dict = None  # статистика по тегам
    health: dict = None  # здоровье памяти
    emotional_trend: str | None = None  # эмоциональный тренд
    radar_items: list = None  # Reply Radar (list[RadarItem])

    def __post_init__(self) -> None:
        if self.waiting_reply is None:
            self.waiting_reply = []
        if self.overdue_commitments is None:
            self.overdue_commitments = []
        if self.today_commitments is None:
            self.today_commitments = []
        if self.bridges is None:
            self.bridges = []
        if self.cross_insights is None:
            self.cross_insights = []
        if self.radar_items is None:
            self.radar_items = []


async def collect_briefing_data(owner_id: int) -> BriefingData:
    """Собирает полные данные для ручного брифинга (/briefing)."""
    result = BriefingData()
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)

        # Активные переписки со статусом waiting_reply
        active = await list_active_conversations(
            session, owner, status="waiting_reply", limit=20
        )
        for conv in active:
            result.waiting_reply.append(
                {"peer_id": conv.peer_id, "unread": conv.unread_count}
            )
            result.unread_total += conv.unread_count

        # Просроченные и сегодняшние обязательства
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        today_end = now.replace(hour=23, minute=59, second=59, microsecond=999999)
        open_commits = await list_open_commitments(session, owner)
        for c in open_commits:
            if c.deadline_at and c.deadline_at < now:
                result.overdue_commitments.append(
                    {"text": c.text[:100], "deadline": str(c.deadline_at)}
                )
            elif c.deadline_at and c.deadline_at <= today_end:
                result.today_commitments.append(
                    {"text": c.text[:100], "deadline": str(c.deadline_at)}
                )

        # Новые факты в памяти за 24 часа
        cutoff = now - timedelta(hours=24)
        memories = await list_memories(session, owner)
        result.recent_memories = sum(
            1 for m in memories if m.created_at and m.created_at >= cutoff
        )

        # Общая статистика памяти
        result.memory_stats = await get_memory_stats(session, owner)

    # Статистика топлива памяти (открывает свою сессию)
    result.fuel_stats = await get_fuel_stats(owner_id)

    # Статистика по временным слоям памяти
    result.layer_stats = await get_layer_stats(owner_id)

    # Статистика по тегам
    from src.core.memory.memory_tagger import get_tag_stats

    result.tag_stats = await get_tag_stats(owner_id)

    # Смысловые мосты между контактами
    from src.core.memory.memory_neighbors import (
        cross_contact_insights,
        find_cross_contact_bridges,
    )

    result.bridges = await find_cross_contact_bridges(owner_id)

    # Кросс-контактные инсайты (общие темы между контактами)
    result.cross_insights = await cross_contact_insights(owner_id)

    # Здоровье памяти
    result.health = await calculate_health_score(owner_id)

    # Эмоциональный тренд
    result.emotional_trend = await compute_emotional_trend(owner_id)

    # Reply Radar: самые срочные для ответа контакты
    result.radar_items = await collect_reply_radar(owner_id, limit=3)

    return result


def format_briefing(data: BriefingData, title: str) -> str:
    """Форматирует полный брифинг в HTML (для /briefing)."""
    lines = [f"<b>📋 {title}</b>", ""]

    if data.health:
        lines.append(format_health_compact(data.health))
        lines.append("")

    if data.unread_total > 0:
        lines.append(
            f"📬 <b>Непрочитанные:</b> {data.unread_total} сообщений "
            f"от {len(data.waiting_reply)} контактов"
        )
    else:
        lines.append("📬 Непрочитанных нет.")

    if data.overdue_commitments:
        lines.append(f"\n🚨 <b>Просрочено ({len(data.overdue_commitments)}):</b>")
        for c in data.overdue_commitments[:5]:
            lines.append(f"• {c['text']}")

    if data.today_commitments:
        lines.append(f"\n📋 <b>Сегодня ({len(data.today_commitments)}):</b>")
        for c in data.today_commitments[:5]:
            lines.append(f"• {c['text']}")

    if data.recent_memories > 0:
        lines.append(f"\n🧠 Новых фактов в памяти: {data.recent_memories}")

    if data.memory_stats:
        total = data.memory_stats.get("total", 0)
        by_sentiment = data.memory_stats.get("by_sentiment", {})
        _positive = by_sentiment.get("positive", 0)
        _negative = by_sentiment.get("negative", 0)
        by_source = data.memory_stats.get("by_source", {})
        by_tier = data.memory_stats.get("by_tier", {})
        weekly_facts = by_source.get("weekly", 0)
        tier_1 = by_tier.get("tier_1", 0)
        tier_2 = by_tier.get("tier_2", 0)
        tier_3 = by_tier.get("tier_3", 0)
        lines.append(
            f"\n🧠 Память: {total} фактов "
            f"({tier_1} эпизодов, {tier_2} недельных, {tier_3} месячных)"
        )
        if weekly_facts:
            lines.append(f"   📊 Из них {weekly_facts} с недельного саммари")

    # Статистика тегов
    if data.tag_stats:
        from src.core.memory.memory_tagger import format_tag_stats

        tag_lines = format_tag_stats(data.tag_stats)
        if tag_lines:
            lines.append("")
            lines.append(tag_lines)

    # Индикатор топлива памяти
    if data.fuel_stats:
        from src.core.memory.memory_fuel import (
            format_depleted_contacts,
            format_fuel_line,
        )

        lines.append("")
        lines.append(format_fuel_line(data.fuel_stats))
        depleted_text = format_depleted_contacts(data.fuel_stats)
        if depleted_text:
            lines.append(depleted_text)

    # Временные слои памяти
    if data.layer_stats and data.layer_stats.get("total", 0) > 0:
        lines.append("")
        lines.append(format_layer_stats(data.layer_stats))

    # Смысловые мосты между контактами
    if data.bridges:
        from src.core.memory.memory_neighbors import format_bridges

        lines.append("")
        lines.append(format_bridges(data.bridges))

    # Кросс-контактные инсайты
    if data.cross_insights:
        from src.core.memory.memory_neighbors import format_cross_insights

        lines.append("")
        lines.append(format_cross_insights(data.cross_insights))

    # Эмоциональный тренд
    if data.emotional_trend:
        lines.append("")
        lines.append(data.emotional_trend)

    # Reply Radar
    if data.radar_items:
        lines.append("")
        lines.append("<b>📡 Кому ответить:</b>")
        for item in data.radar_items:
            lines.append(
                f"• <b>{item.contact_name}</b> — ждёт {item.waiting_hours:.0f}ч [{item.score}]"
            )
        lines.append("<i>/today — открыть полный пульт</i>")

    if data.health:
        lines.append("")
        lines.append(format_health(data.health))

    lines.append("\n💬 <i>/threads — просмотреть переписки</i>")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Утренний дайджест — новый компактный формат для 9:00
# ═══════════════════════════════════════════════════════════════════


async def _collect_morning_digest(owner_id: int) -> str:
    """Собирает утренний дайджест: вчера, неотвеченные, дедлайны, здоровье, план."""
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_id)
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday_start = today_start - timedelta(days=1)
        seven_days_ago = now - timedelta(days=7)
        two_days_ahead = now + timedelta(hours=48)

        lines = ["☀️ Доброе утро!"]

        # ── 1. Вчерашняя статистика ────────────────────────────────
        dialogs_r = await session.execute(
            select(func.count(func.distinct(Message.peer_id))).where(
                Message.user_id == owner.id,
                Message.date >= yesterday_start,
                Message.date < today_start,
            )
        )
        yday_dialogs = dialogs_r.scalar_one() or 0

        conflicts_r = await session.execute(
            select(func.count())
            .select_from(Memory)
            .where(
                Memory.user_id == owner.id,
                Memory.sentiment == "negative",
                Memory.created_at >= yesterday_start,
                Memory.created_at < today_start,
            )
        )
        yday_conflicts = conflicts_r.scalar_one() or 0

        commits_r = await session.execute(
            select(func.count())
            .select_from(Commitment)
            .where(
                Commitment.user_id == owner.id,
                Commitment.created_at >= yesterday_start,
                Commitment.created_at < today_start,
            )
        )
        yday_commits = commits_r.scalar_one() or 0

        lines.append(
            f"📊 Вчера: {yday_dialogs} диалогов, "
            f"{yday_conflicts} конфликт, "
            f"{yday_commits} обещания."
        )
        lines.append("")

        # ── 2. Неотвеченные сообщения ──────────────────────────────
        unanswered_r = await session.execute(
            select(ConversationState)
            .where(
                ConversationState.user_id == owner.id,
                ConversationState.last_incoming_at >= seven_days_ago,
                or_(
                    ConversationState.last_outgoing_at.is_(None),
                    ConversationState.last_incoming_at
                    > ConversationState.last_outgoing_at,
                ),
                ConversationState.status != "archived",
            )
            .order_by(desc(ConversationState.last_incoming_at))
            .limit(10)
        )
        unanswered = list(unanswered_r.scalars().all())

        needs_reply_names: list[str] = []
        for conv in unanswered:
            if conv.last_incoming_at is None:
                continue
            hours = (now - conv.last_incoming_at).total_seconds() / 3600
            urgency = "🔴" if hours > 48 else "🟡"

            cr = await session.execute(
                select(Contact.display_name).where(
                    Contact.user_id == owner.id,
                    Contact.peer_id == conv.peer_id,
                )
            )
            name = cr.scalar_one_or_none() or str(conv.peer_id)

            # Последний входящий текст
            mr = await session.execute(
                select(Message.text)
                .where(
                    Message.user_id == owner.id,
                    Message.peer_id == conv.peer_id,
                    Message.is_outgoing.is_(False),
                )
                .order_by(desc(Message.date))
                .limit(1)
            )
            last_text = mr.scalar_one_or_none() or ""
            snippet = last_text[:50].replace("\n", " ") if last_text else ""

            days = int(hours / 24)
            if days >= 1:
                time_str = f"ждёт ответ {days} дн."
            else:
                time_str = f"ждёт ответ {int(hours)} ч."

            line = f"{urgency} {name}: {time_str}"
            if snippet:
                line += f" Последнее: «{snippet}»"
            lines.append(line)
            needs_reply_names.append(name)

        if unanswered:
            lines.append("")

        # ── 3. Дедлайны в ближайшие 48ч ────────────────────────────
        dl_r = await session.execute(
            select(Commitment)
            .where(
                Commitment.user_id == owner.id,
                Commitment.status == "open",
                Commitment.deadline_at.is_not(None),
                Commitment.deadline_at >= now,
                Commitment.deadline_at <= two_days_ahead,
            )
            .order_by(Commitment.deadline_at.asc())
            .limit(10)
        )
        upcoming = list(dl_r.scalars().all())

        deadline_names: list[str] = []
        for c in upcoming:
            cr = await session.execute(
                select(Contact.display_name).where(
                    Contact.user_id == owner.id,
                    Contact.peer_id == c.peer_id,
                )
            )
            name = cr.scalar_one_or_none() or str(c.peer_id)

            dl = c.deadline_at
            if dl and dl.tzinfo is not None:
                dl = dl.replace(tzinfo=None)

            if dl:
                hours_left = (dl - now).total_seconds() / 3600
                if hours_left <= 24:
                    time_str = "дедлайн сегодня"
                else:
                    time_str = "дедлайн завтра"
            else:
                time_str = ""

            text = (c.text or "")[:80]
            lines.append(f"🟡 {name}: {time_str} — {text}")
            deadline_names.append(name)

        if upcoming:
            lines.append("")

        # ── 4. Здоровые контакты ───────────────────────────────────
        healthy_lines = await _collect_healthy(
            session, owner, yesterday_start, today_start, seven_days_ago
        )
        lines.extend(healthy_lines)

        if healthy_lines:
            lines.append("")

        # ── 5. План на сегодня ─────────────────────────────────────
        plan: list[str] = []
        if needs_reply_names:
            plan.append(f"ответить {', '.join(needs_reply_names[:3])}")
        if deadline_names:
            plan.append(f"проверить: {', '.join(deadline_names[:3])}")

        if plan:
            lines.append(f"📋 На сегодня: {'; '.join(plan)}")

        return "\n".join(lines)


async def _collect_healthy(
    session, owner, yesterday_start, today_start, seven_days_ago
) -> list[str]:
    """Собирает 💚-строки для контактов с health > 80 и недавней активностью."""
    rows_r = await session.execute(
        select(
            Contact.peer_id,
            Contact.display_name,
            func.count(Message.id).label("msg_total"),
            func.max(Message.date).label("last_date"),
            func.sum(case((Message.is_outgoing.is_(True), 1), else_=0)).label(
                "outgoing"
            ),
        )
        .join(
            Message,
            and_(
                Message.user_id == Contact.user_id,
                Message.peer_id == Contact.peer_id,
            ),
        )
        .where(
            Contact.user_id == owner.id,
            Message.date >= seven_days_ago,
        )
        .group_by(Contact.peer_id, Contact.display_name)
    )

    rows = list(rows_r.all())

    # ── Batch load yesterday message counts for all peers ──────────
    all_peer_ids = [r[0] for r in rows]
    if all_peer_ids:
        yday_r = await session.execute(
            select(Message.peer_id, func.count().label("cnt"))
            .where(
                Message.user_id == owner.id,
                Message.peer_id.in_(all_peer_ids),
                Message.date >= yesterday_start,
                Message.date < today_start,
            )
            .group_by(Message.peer_id)
        )
        yday_by_peer = {r[0]: r[1] for r in yday_r.all()}
    else:
        yday_by_peer = {}

    result: list[str] = []
    for row in rows:
        peer_id, name, msg_total, last_date, outgoing = row
        outgoing = outgoing or 0

        if last_date:
            if last_date.tzinfo is None:
                last_date = last_date.replace(tzinfo=timezone.utc)
            days_gap = (datetime.now(timezone.utc) - last_date).days
        else:
            days_gap = 365

        reply_ratio = outgoing / max(msg_total, 1)

        # Упрощённая формула health_score (как в health_score.py)
        score = 100.0
        score -= min(days_gap / 7.0 * 10.0, 60.0)
        if msg_total < 10:
            score -= 10.0
        if 0.3 <= reply_ratio <= 0.7:
            score += 10.0
        elif reply_ratio > 0.9:
            score -= 15.0
        elif reply_ratio < 0.1 and msg_total > 5:
            score -= 20.0
        score = max(0.0, min(100.0, round(score)))

        if score >= 80:
            yday_msgs = yday_by_peer.get(peer_id, 0)
            result.append(f"💚 С {name} всё хорошо, {yday_msgs} сообщений вчера")

    return result[:5]


# ═══════════════════════════════════════════════════════════════════
# Scheduler loop — только утренний брифинг в 9:00
# ═══════════════════════════════════════════════════════════════════


async def proactive_briefing_loop(owner_id: int) -> None:
    """Фоновый цикл: утренний дайджест в 9:00–9:05 по tz владельца."""
    from src.core.infra.timeutil import now_in_tz

    while True:
        try:
            async with get_session() as session:
                owner = await get_or_create_user(session, owner_id)
                tz_name = (
                    owner.settings.timezone
                    if owner.settings and owner.settings.timezone
                    else "UTC"
                )

            now_tz = now_in_tz(tz_name)
            if now_tz.hour == 9 and now_tz.minute < 5:
                # Вычислить начало часового слота в tz пользователя, затем в UTC
                slot_start_tz = now_tz.replace(
                    hour=9, minute=0, second=0, microsecond=0
                )
                slot_start_utc = slot_start_tz.astimezone(timezone.utc).replace(
                    tzinfo=None
                )

                async with get_session() as session:
                    owner = await get_or_create_user(session, owner_id)
                    if owner.settings.proactive_last_sent == slot_start_utc:
                        await asyncio.sleep(settings.proactive_briefing_check_sec)
                        continue
                    owner.settings.proactive_last_sent = slot_start_utc
                    await session.commit()

                text = await _collect_morning_digest(owner_id)
                await notification_queue.enqueue(
                    topic="briefing",
                    text=text,
                    priority=Notification.PRIORITY_MEDIUM,
                    category="morning",
                )
                await asyncio.sleep(3600)  # не повторять в этот час

            await asyncio.sleep(settings.proactive_briefing_check_sec)
        except Exception:
            logger.exception("Briefing loop error")
            await asyncio.sleep(settings.proactive_briefing_check_sec)


task_manager.register(
    "proactive-briefing", partial(proactive_briefing_loop, settings.owner_telegram_id)
)
