"""Отслеживание вопросов, на которые модель не смогла ответить."""

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# In-memory хранилище: {telegram_id: [{"question": ..., "context": ..., "ts": ...}]}
_pending: dict[int, list[dict[str, Any]]] = {}
_pending_lock = asyncio.Lock()

_PENDING_TTL = 7 * 86400  # 7 дней
_save_counter: int = 0
_CLEANUP_EVERY_N = 100  # cleanup раз в ~100 вызовов


def _cleanup_stale_pending() -> None:
    """Удаляет pending-записи старше _PENDING_TTL."""
    now = time.time()
    cutoff = now - _PENDING_TTL
    for uid in list(_pending.keys()):
        _pending[uid] = [q for q in _pending[uid] if q.get("ts", 0) > cutoff]
        if not _pending[uid]:
            del _pending[uid]


async def save_pending(telegram_id: int, question: str, context: str = "") -> None:
    """Сохраняет вопрос, на который не нашлось ответа."""
    global _save_counter
    async with _pending_lock:
        _pending.setdefault(telegram_id, []).append(
            {
                "question": question[:500],
                "context": context[:200],
                "ts": time.time(),
            }
        )
        # Ограничиваем 20 вопросами на пользователя
        if len(_pending[telegram_id]) > 20:
            _pending[telegram_id] = _pending[telegram_id][-20:]

        _save_counter += 1
        if _save_counter % _CLEANUP_EVERY_N == 0:
            _cleanup_stale_pending()


async def get_pending(telegram_id: int) -> list[dict[str, Any]]:
    """Возвращает список неотвеченных вопросов."""
    async with _pending_lock:
        return _pending.get(telegram_id, [])


async def clear_pending(telegram_id: int) -> None:
    """Очищает после успешного ответа."""
    async with _pending_lock:
        _pending.pop(telegram_id, None)
