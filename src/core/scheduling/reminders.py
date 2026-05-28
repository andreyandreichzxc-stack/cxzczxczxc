"""Напоминания о Commitment'ах: пинги об overdue и о приближении дедлайна."""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import or_, select

from src.config import settings
from src.core.scheduling.notification_queue import notification_queue
from src.db.models import Notification
from src.core.infra.timeutil import fmt_local
from src.db.models import Commitment
from src.db.repo import get_or_create_user
from src.db.session import get_session


logger = logging.getLogger(__name__)


REMINDER_TICK_SECONDS = 300


async def _check_once(owner_telegram_id: int) -> None:
    async with get_session() as session:
        owner = await get_or_create_user(session, owner_telegram_id)
        s = owner.settings
        if not s.reminders_enabled:
            return

        tz_name = s.timezone
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        lead_hours = max(0, int(s.reminder_lead_hours))
        soon = now + timedelta(hours=lead_hours)

        # Allow re-reminding after 24h cooldown
        cutoff = now - timedelta(hours=24)
        result = await session.execute(
            select(Commitment).where(
                Commitment.user_id == owner.id,
                Commitment.status == "open",
                Commitment.deadline_at.is_not(None),
                or_(
                    Commitment.last_reminded_at.is_(None),
                    Commitment.last_reminded_at < cutoff.replace(tzinfo=None),
                ),
            )
        )
        open_items = list(result.scalars().all())

        if not open_items:
            return

        to_remind: list[tuple[Commitment, str]] = []
        for c in open_items:
            d = c.deadline_at
            if d is None:
                continue
            if d < now and s.reminder_overdue_enabled:
                to_remind.append((c, "overdue"))
            elif now <= d <= soon and lead_hours > 0:
                to_remind.append((c, "lead"))

        if not to_remind:
            return

        for commitment, reason in to_remind:
            who = (
                "Я"
                if commitment.direction == "mine"
                else (commitment.peer_name or "Они")
            )
            d = fmt_local(commitment.deadline_at, tz_name)
            if reason == "overdue":
                text = f"⏰ <b>Просрочено</b>\n<b>{who}</b>: {commitment.text}\nСрок был: {d}"
            else:
                text = (
                    f"⏳ <b>Скоро дедлайн</b>\n<b>{who}</b>: {commitment.text}\nДо: {d}"
                )
            await notification_queue.enqueue(
                topic="reminders",
                text=text,
                priority=Notification.PRIORITY_HIGH,
                category=reason,  # "overdue" или "lead"
            )

        # помечаем когда отправили напоминание (не меняем status!)
        for c, _ in to_remind:
            c.last_reminded_at = datetime.now(timezone.utc)
        await session.commit()


from src.core.infra.task_manager import task_manager


@task_manager.task("reminders-loop")
async def reminders_loop() -> None:
    while True:
        try:
            await _check_once(settings.owner_telegram_id)
        except Exception:
            logger.exception("reminders tick failed")
        await asyncio.sleep(REMINDER_TICK_SECONDS)
