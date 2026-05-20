import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from src.core.memory_fuel import get_fuel_stats
from src.core.memory_health import (
    calculate_health_score,
    compute_emotional_trend,
    format_health,
    format_health_compact,
)
from src.core.notification_queue import notification_queue
from src.db.models import Notification
from src.core.reply_radar import collect_reply_radar
from src.core.temporal_layers import format_layer_stats, get_layer_stats
from src.db.repo import (
    get_memory_stats,
    get_or_create_user,
    list_active_conversations,
    list_memories,
    list_open_commitments,
)
from src.config import settings
from src.db.session import get_session

logger = logging.getLogger(__name__)


@dataclass
class BriefingData:
    urgent_count: int = 0
    unread_total: int = 0
    waiting_reply: list = None  # list[dict]
    overdue_commitments: list = None  # list[dict]
    today_commitments: list = None  # list[dict]
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
    """Собирает данные для брифинга: срочные сообщения, ожидающие ответа, обязательства."""
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
        now = datetime.now(timezone.utc)
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
    from src.core.memory_tagger import get_tag_stats

    result.tag_stats = await get_tag_stats(owner_id)

    # Смысловые мосты между контактами
    from src.core.memory_neighbors import (
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
    """Форматирует брифинг в HTML."""
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
        positive = by_sentiment.get("positive", 0)
        negative = by_sentiment.get("negative", 0)
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
        from src.core.memory_tagger import format_tag_stats

        tag_lines = format_tag_stats(data.tag_stats)
        if tag_lines:
            lines.append("")
            lines.append(tag_lines)

    # Индикатор топлива памяти
    if data.fuel_stats:
        from src.core.memory_fuel import format_depleted_contacts, format_fuel_line

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
        from src.core.memory_neighbors import format_bridges

        lines.append("")
        lines.append(format_bridges(data.bridges))

    # Кросс-контактные инсайты
    if data.cross_insights:
        from src.core.memory_neighbors import format_cross_insights

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


async def proactive_briefing_loop(owner_id: int) -> None:
    """Фоновый цикл: утренний брифинг в 9:00, вечерний в 21:00 по tz владельца."""
    from src.core.timeutil import now_in_tz

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
            hour = now.hour

            if hour == 9:
                data = await collect_briefing_data(owner_id)
                text = format_briefing(data, "☀️ Утренний брифинг")
                await notification_queue.enqueue(
                    topic="briefing",
                    text=text,
                    priority=Notification.PRIORITY_MEDIUM,
                    category="morning",
                )
                await asyncio.sleep(3600)  # не повторять в этот час
            elif hour == 21:
                data = await collect_briefing_data(owner_id)
                text = format_briefing(data, "🌙 Вечерний брифинг")
                await notification_queue.enqueue(
                    topic="briefing",
                    text=text,
                    priority=Notification.PRIORITY_MEDIUM,
                    category="evening",
                )
                await asyncio.sleep(3600)

            await asyncio.sleep(
                settings.proactive_briefing_check_sec
            )  # проверка каждые 5 минут
        except Exception:
            logger.exception("Briefing loop error")
            await asyncio.sleep(settings.proactive_briefing_check_sec)
