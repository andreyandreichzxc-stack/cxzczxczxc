"""Асинхронная очередь для фоновой обработки памяти.

Позволяет вынести сохранение, извлечение и тегирование фактов
из основного потока обработки сообщений в фоновый worker.
"""

import asyncio
import logging
from dataclasses import dataclass

from src.db.repo import get_or_create_user
from src.db.session import get_session
from src.llm.router import build_provider

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
_worker_task: asyncio.Task | None = None
_worker_lock: asyncio.Lock = asyncio.Lock()


async def _worker() -> None:
    """Фоновый обработчик очереди.

    Бесконечный цикл: забирает задание из очереди и выполняет.
    При крахе одной задачи не падает — логирует и идёт дальше.
    """
    while True:
        try:
            job: MemoryJob = await _queue.get()
            await _process_job(job)
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("Memory queue worker error")
        finally:
            _queue.task_done()


async def _process_job(job: MemoryJob) -> None:
    """Выполнить одно задание."""
    async with get_session() as session:
        owner = await get_or_create_user(session, job.telegram_id)

        if job.job_type == "save":
            await _handle_save(session, owner, job)
        elif job.job_type == "extract":
            await _handle_extract(session, owner, job)
        elif job.job_type == "tag":
            await _handle_tag(session, owner, job)
        else:
            logger.warning("Unknown memory job type: %s", job.job_type)


async def _handle_save(session, owner, job: MemoryJob) -> None:
    """Сохранить готовые факты (job_type='save')."""
    from src.db.repo import add_memory, link_memories
    from src.core.actions.vector_store import get_vector_store

    saved_memories: list = []
    for fact_data in job.facts or []:
        try:
            mem = await add_memory(
                session,
                owner,
                fact=fact_data.get("fact", ""),
                contact_id=job.contact_id,
                sentiment=fact_data.get("sentiment"),
                source=fact_data.get("source", "chat"),
                importance=fact_data.get("importance", 0.5),
                decay_rate=fact_data.get("decay_rate", 0.07),
                memory_type=fact_data.get("memory_type"),
                embedding=fact_data.get("embedding"),
                vector_store_obj=get_vector_store()
                if fact_data.get("embedding")
                else None,
            )
            if mem:
                saved_memories.append(mem)
            await session.commit()
        except Exception:
            await session.rollback()
            logger.exception(
                "Failed to save fact for user %d, skipping", job.telegram_id
            )

    # Сохраняем связи между фактами, указанные LLM (relation_type / relation_to_index)
    for i, fact_data in enumerate(job.facts or []):
        if i >= len(saved_memories):
            continue
        relation_type = fact_data.get("relation_type")
        relation_to_index = fact_data.get("relation_to_index")
        if relation_type and relation_to_index is not None:
            target_idx = int(relation_to_index)
            if 0 <= target_idx < len(saved_memories):
                await link_memories(
                    session,
                    owner,
                    source_id=saved_memories[i].id,
                    target_id=saved_memories[target_idx].id,
                    relation_type=relation_type,
                    weight=0.9,
                )

    # --- Persona auto-rebuild: check if enough new personal facts ---
    try:
        from src.core.memory.persona_pipeline import maybe_rebuild_persona

        # Only trigger if we saved personal/self-facts
        has_personal_facts = any(
            fact_data.get("memory_type") in {"personal", "preference"}
            for fact_data in (job.facts or [])
        )
        if has_personal_facts:
            await maybe_rebuild_persona(session, owner)
    except Exception:
        logger.debug("Persona auto-rebuild skipped (non-critical)", exc_info=True)

    await session.commit()
    logger.debug(
        "Background saved %d facts for user %d", len(job.facts or []), job.telegram_id
    )

    # ── Invalidate contact memory digest ────────────────────────────
    if job.contact_id is not None:
        try:
            from src.core.contacts.contact_memory_digest import (
                invalidate_contact_digest,
            )

            await invalidate_contact_digest(job.contact_id)
        except Exception:
            logger.debug(
                "Failed to invalidate digest for peer %d",
                job.contact_id,
                exc_info=True,
            )


async def _handle_extract(session, owner, job: MemoryJob) -> None:
    """Извлечь и сохранить факты из текста переписки (job_type='extract')."""
    from src.core.memory.memory_extractor import extract_and_save_memories

    provider = await build_provider(session, owner)
    if provider is None:
        logger.warning("No provider for extract job uid=%d", job.telegram_id)
        return

    # Получить объект Contact по peer_id
    contact = None
    if job.contact_id is not None:
        from sqlalchemy import select
        from src.db.models import Contact

        result = await session.execute(
            select(Contact).where(
                Contact.user_id == owner.id,
                Contact.peer_id == job.contact_id,
            )
        )
        contact = result.scalar_one_or_none()

    # Вызвать extract_and_save_memories — она сделает LLM-вызов и
    # поставит задачу на сохранение в ту же очередь (job_type='save')
    count = await extract_and_save_memories(
        provider,
        job.telegram_id,
        contact,
        messages=[],
        transcript=job.messages_text,
    )

    # --- Persona auto-rebuild ---
    try:
        from src.core.memory.persona_pipeline import maybe_rebuild_persona

        await maybe_rebuild_persona(session, owner)
    except Exception:
        logger.debug("Persona auto-rebuild skipped (non-critical)", exc_info=True)

    logger.debug(
        "Background extracted %d facts for user %d (contact %s)",
        count,
        job.telegram_id,
        job.contact_id,
    )


async def _handle_tag(session, owner, job: MemoryJob) -> None:
    """Протегировать нетэгированные факты (job_type='tag')."""
    from src.core.memory.memory_tagger import tag_new_fact
    from src.db.repo import list_memories

    provider = await build_provider(session, owner)
    if provider is None:
        logger.warning("No provider for tag job uid=%d", job.telegram_id)
        return

    memories = await list_memories(session, owner)
    tagged = 0
    MAX_TAG_PER_CYCLE = 30
    for mem in memories:
        if tagged >= MAX_TAG_PER_CYCLE:
            logger.debug("_handle_tag: hit limit %d, stopping", MAX_TAG_PER_CYCLE)
            break
        if not mem.tags:
            try:
                await tag_new_fact(provider, session, mem.id)
                await session.commit()
                tagged += 1
            except (ValueError, AttributeError, ConnectionError, OSError):
                await session.rollback()
                logger.exception("Tagging failed for memory %d", mem.id)
    logger.debug(
        "Background tagging done for user %d (%d tagged)", job.telegram_id, tagged
    )


async def start_worker() -> asyncio.Task:
    """Запустить фонового worker'а (если ещё не запущен).

    Вызывается при старте приложения (main.py).
    """
    global _worker_task
    async with _worker_lock:
        if _worker_task is None or _worker_task.done():
            _worker_task = asyncio.create_task(_worker(), name="memory-queue-worker")
        return _worker_task


async def enqueue(job: MemoryJob) -> None:
    """Добавить задание в очередь (с таймаутом 10с).

    Если очередь переполнена — отправитель ждёт до 10 секунд,
    после чего задание отбрасывается с error-логом.
    """
    try:
        await asyncio.wait_for(_queue.put(job), timeout=10.0)
    except asyncio.TimeoutError:
        logger.error("Memory queue stuck, dropping job: %s", job.job_type)


async def stop_worker() -> None:
    """Остановить фонового worker'а (graceful shutdown)."""
    global _worker_task
    async with _worker_lock:
        if _worker_task and not _worker_task.done():
            _worker_task.cancel()
            try:
                await _worker_task
            except asyncio.CancelledError:
                pass
            _worker_task = None
            logger.info("Memory queue worker stopped")
