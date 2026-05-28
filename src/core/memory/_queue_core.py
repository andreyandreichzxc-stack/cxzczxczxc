"""Core queue primitives for memory background processing.

Houses MemoryJob, the async queue, and enqueue() — extracted from
memory_queue.py to break the memory_queue ↔ memory_extractor cycle.
All other modules import from here instead of from each other.
"""

import asyncio
import logging
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class MemoryJob:
    """Задача на фоновую обработку памяти.

    telegram_id — Telegram ID владельца (message.from_user.id).
    contact_id — Contact.peer_id (Telegram peer_id собеседника).
    facts — список словарей с фактами для сохранения.
    messages_text — текст переписки для извлечения фактов.
    job_type — тип задачи: save | extract | tag.
    """

    telegram_id: int
    contact_id: int | None = None
    facts: list[dict] | None = None
    messages_text: str = ""
    job_type: str = "save"


# Очередь заданий (maxsize=100 — защита от переполнения памяти)
_queue: asyncio.Queue[MemoryJob] = asyncio.Queue(maxsize=100)


async def enqueue(job: MemoryJob) -> None:
    """Добавить задание в очередь (с таймаутом 10с).

    Если очередь переполнена — отправитель ждёт до 10 секунд,
    после чего задание отбрасывается с error-логом.
    """
    try:
        await asyncio.wait_for(_queue.put(job), timeout=10.0)
    except asyncio.TimeoutError:
        logger.error("Memory queue stuck, dropping job: %s", job.job_type)
