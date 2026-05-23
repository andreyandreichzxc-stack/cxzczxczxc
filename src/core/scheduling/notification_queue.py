"""Асинхронная очередь уведомлений с группировкой по topic + category."""

import asyncio
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Optional

from sqlalchemy import select, update

from src.core.infra.notifier import notifier
from src.db.models import Notification
from src.db.session import SessionLocal

if TYPE_CHECKING:
    from aiogram.types import InlineKeyboardMarkup


logger = logging.getLogger(__name__)


class NotificationQueue:
    """
    Асинхронная очередь уведомлений с группировкой по topic + category.

    Правила:
    - Уведомления одной topic в окне 5 минут — группируются в один batch
    - Приоритет: CRITICAL → немедленно (без очереди)
    - HIGH/MEDIUM/LOW → группируются, отправляются раз в 60 секунд
    - Максимум 10 уведомлений в одном batch-сообщении
    - TTL 24 часа для недоставленных
    - Уведомления с reply_markup (inline-клавиатуры) отправляются немедленно
    """

    def __init__(self) -> None:
        self._window_seconds = 300  # 5 минут
        self._flush_interval = 60  # проверка раз в минуту
        self._max_batch_size = 10
        self._ttl_hours = 24
        self._loop_task: Optional[asyncio.Task] = None

    async def enqueue(
        self,
        topic: str,
        text: str,
        priority: int = Notification.PRIORITY_MEDIUM,
        category: str = "",
        metadata: dict | None = None,
        reply_markup: "InlineKeyboardMarkup | None" = None,
    ) -> int:
        """
        Добавить уведомление в очередь.

        CRITICAL (priority=0) или уведомления с reply_markup —
        отправляются немедленно, минуя очередь.

        Возвращает notification_id (0 для немедленно отправленных).
        """
        # Уведомления с inline-клавиатурами — немедленная отправка
        if reply_markup is not None:
            await notifier.notify(text, reply_markup=reply_markup)
            return 0

        # Критические — немедленная отправка
        if priority == Notification.PRIORITY_CRITICAL:
            await notifier.notify(text)
            return 0

        async with SessionLocal() as session:
            notif = Notification(
                topic=topic,
                priority=priority,
                category=category or topic,
                text=text,
                metadata_json=metadata or {},
            )
            session.add(notif)
            await session.commit()
            await session.refresh(notif)
            return notif.id

    async def flush(self) -> int:
        """
        Группирует и отправляет непрочитанные уведомления.
        Группировка: все уведомления одной topic за последние window_seconds.
        Возвращает количество обработанных.
        """
        async with SessionLocal() as session:
            result = await session.execute(
                select(Notification)
                .where(
                    Notification.flushed_at.is_(None),
                )
                .order_by(
                    Notification.topic, Notification.priority, Notification.created_at
                )
                .with_for_update()
            )
            pending = list(result.scalars().all())

            if not pending:
                return 0

            # Разделяем: свежие (в окне) — группируем, старые — отправляем по одному
            window_start = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
                seconds=self._window_seconds
            )
            fresh = [n for n in pending if n.created_at >= window_start]
            stale = [n for n in pending if n.created_at < window_start]

            # Группировка свежих по (topic, priority_bucket)
            groups: dict[str, list[Notification]] = defaultdict(list)
            for n in fresh:
                if n.priority <= Notification.PRIORITY_HIGH:
                    bucket = "high"
                elif n.priority == Notification.PRIORITY_MEDIUM:
                    bucket = "medium"
                else:
                    bucket = "low"
                key = f"{n.topic}:{bucket}"
                groups[key].append(n)

            # Старые — каждое в своей группе для немедленной отправки
            for n in stale:
                key = f"{n.topic}:stale_{n.id}"
                groups[key] = [n]

            batch_id = uuid.uuid4().hex[:12]
            total_flushed = 0

            for key, group in groups.items():
                topic = key.split(":")[0]
                batch = group[: self._max_batch_size]
                text = self._format_batch(topic, batch)

                try:
                    await notifier.notify(text)
                    total_flushed += len(batch)
                except Exception:
                    logger.exception("Failed to send batch for topic %s", topic)
                    continue

                # Помечаем отправленные как flushed (в той же транзакции, что и SELECT FOR UPDATE)
                ids = [n.id for n in batch]
                await session.execute(
                    update(Notification)
                    .where(Notification.id.in_(ids))
                    .values(
                        flushed_at=datetime.now(timezone.utc).replace(tzinfo=None),
                        batch_id=batch_id,
                    )
                )

                # Остаток (>max_batch_size) — оставляем на следующий flush
                # (они остаются с flushed_at=NULL и будут обработаны в следующей итерации)

            await session.commit()
            return total_flushed

    def _format_batch(self, topic: str, notifications: list[Notification]) -> str:
        """Форматирует сгруппированные уведомления в одно сообщение."""
        count = len(notifications)

        # Группируем по приоритету внутри batch
        by_priority: dict[int, list[Notification]] = defaultdict(list)
        for n in notifications:
            by_priority[n.priority].append(n)

        priority_emoji = {
            Notification.PRIORITY_CRITICAL: "🔴",
            Notification.PRIORITY_HIGH: "🟠",
            Notification.PRIORITY_MEDIUM: "🟡",
            Notification.PRIORITY_LOW: "🟢",
        }
        _priority_label = {
            Notification.PRIORITY_CRITICAL: "Критическое",
            Notification.PRIORITY_HIGH: "Важное",
            Notification.PRIORITY_MEDIUM: "Обычное",
            Notification.PRIORITY_LOW: "Инфо",
        }

        _topic_ru: dict[str, str] = {
            "system": "система",
            "digest": "дайджест",
            "news": "новости",
            "reminder": "напоминания",
            "task_manager": "задачи",
            "skills": "навыки",
            "memory": "память",
            "contacts": "контакты",
        }

        # Определяем доминирующий приоритет для заголовка
        _dominant = min(by_priority.keys())

        # Собираем разные темы внутри группы
        topic_set: set[str] = {n.category or n.topic for n in notifications}

        lines = [
            f"📬 <b>Сводка</b> ({len(topic_set)} тем, {count} уведомлений)",
            "━" * 28,
        ]

        for prio in sorted(by_priority.keys()):
            items = by_priority[prio]
            emoji = priority_emoji.get(prio, "⚪")
            sub_topic = _topic_ru.get(
                items[0].category or items[0].topic, items[0].category or items[0].topic
            )
            lines.append(f"{emoji} <b>{sub_topic}</b> ({len(items)})")
            for item in items:
                # Обрезаем длинный текст
                short = item.text[:200]
                if len(item.text) > 200:
                    short += "…"
                lines.append(f"• {short}")
            lines.append("")

        return "\n".join(lines)

    async def flush_loop(self) -> None:
        """Бесконечный цикл: flush() + периодическая очистка."""
        _cleanup_counter = 0
        while True:
            try:
                flushed = await self.flush()
                if flushed > 0:
                    logger.info("Flushed %d notifications", flushed)
            except Exception:
                logger.exception("NotificationQueue flush error")

            # Очистка просроченных — раз в час (60 итераций при интервале 60с)
            _cleanup_counter += 1
            if _cleanup_counter >= 60:
                _cleanup_counter = 0
                try:
                    cleaned = await self.cleanup_expired()
                    if cleaned > 0:
                        logger.info("Cleaned %d expired notifications", cleaned)
                except Exception:
                    logger.exception("NotificationQueue cleanup error")

            await asyncio.sleep(self._flush_interval)

    def start(self) -> None:
        """Запустить фоновый цикл (идемпотентен)."""
        if self._loop_task is not None and not self._loop_task.done():
            logger.warning("NotificationQueue already running")
            return
        self._loop_task = asyncio.create_task(self.flush_loop())
        logger.info("NotificationQueue started (flush every %ds)", self._flush_interval)

    async def stop(self) -> None:
        """Остановить фоновый цикл."""
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None

    async def cleanup_expired(self) -> int:
        """Удаляет уведомления старше TTL (включая отправленные). Возвращает количество удалённых."""
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
            hours=self._ttl_hours
        )
        async with SessionLocal() as session:
            result = await session.execute(
                select(Notification).where(
                    Notification.created_at < cutoff,
                )
            )
            expired = list(result.scalars().all())
            for n in expired:
                await session.delete(n)
            await session.commit()
            return len(expired)


# Глобальный синглтон (заменяет прямой вызов notifier.notify)
notification_queue = NotificationQueue()
