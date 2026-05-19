import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from src.core.notifier import notifier
from src.db.repo import (
    get_memory_stats,
    get_or_create_user,
    list_active_conversations,
    list_memories,
    list_open_commitments,
)
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

    def __post_init__(self) -> None:
        if self.waiting_reply is None:
            self.waiting_reply = []
        if self.overdue_commitments is None:
            self.overdue_commitments = []
        if self.today_commitments is None:
            self.today_commitments = []


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

    return result


def format_briefing(data: BriefingData, title: str) -> str:
    """Форматирует брифинг в HTML."""
    lines = [f"<b>📋 {title}</b>", ""]
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
        lines.append(
            f"\n🧠 Память: {total} фактов "
            f"({positive} позитивных, {negative} негативных)"
        )

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
                await notifier.notify(text)
                await asyncio.sleep(3600)  # не повторять в этот час
            elif hour == 21:
                data = await collect_briefing_data(owner_id)
                text = format_briefing(data, "🌙 Вечерний брифинг")
                await notifier.notify(text)
                await asyncio.sleep(3600)

            await asyncio.sleep(300)  # проверка каждые 5 минут
        except Exception as e:
            logger.error("Briefing loop error: %s", e)
            await asyncio.sleep(600)
