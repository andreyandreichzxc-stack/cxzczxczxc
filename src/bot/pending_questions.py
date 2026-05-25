"""Pending question queue — accumulate questions during async operations."""

from __future__ import annotations

import asyncio
import logging

from src.db.session import get_session

logger = logging.getLogger(__name__)

_pending: dict[int, list[str]] = {}  # telegram_id → questions
_lock = asyncio.Lock()


async def add_question(telegram_id: int, question: str) -> None:
    # In-memory (fast)
    async with _lock:
        _pending.setdefault(telegram_id, []).append(question)
    # DB (persistent)
    try:
        async with get_session() as session:
            from src.db.repo import add_pending_question, get_or_create_user

            owner = await get_or_create_user(session, telegram_id)
            await add_pending_question(session, owner.id, question)
    except Exception:
        logger.debug("Failed to persist pending question", exc_info=True)


async def get_pending(telegram_id: int) -> list[str]:
    questions: list[str] = []
    # DB first — load any that survived restart (safe: pop happens after)
    try:
        async with get_session() as session:
            from src.db.repo import get_pending_questions, get_or_create_user

            owner = await get_or_create_user(session, telegram_id)
            db_questions = await get_pending_questions(session, owner.id)
            questions.extend(db_questions)
    except Exception:
        logger.debug("Failed to load pending questions from DB", exc_info=True)
    # In-memory (pop after DB so questions are not lost on DB failure)
    async with _lock:
        questions.extend(_pending.pop(telegram_id, []))
    return questions


async def has_pending(telegram_id: int) -> bool:
    """Проверяет наличие ожидающих вопросов в памяти и в БД."""
    async with _lock:
        if _pending.get(telegram_id):
            return True
    # Проверяем также БД (могли остаться после рестарта)
    try:
        async with get_session() as session:
            from src.db.repo import get_or_create_user

            owner = await get_or_create_user(session, telegram_id)
            from sqlalchemy import select, func

            from src.db.models import PendingQuestion

            r = await session.execute(
                select(func.count())
                .select_from(PendingQuestion)
                .where(PendingQuestion.owner_id == owner.id)
            )
            return r.scalar_one() > 0
    except Exception:
        logger.debug("Failed to check pending questions in DB", exc_info=True)
        return False
