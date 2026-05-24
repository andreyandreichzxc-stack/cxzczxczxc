"""
Progress tracker for long-running operations.
Provides real-time message updates via aiogram Message.edit_text().

Usage::

    from src.core.infra.progress import progress_tracker

    async for item in progress_tracker(
        message, len(contacts), contacts,
        item_name_fn=lambda c: c.display_name,
        prefix="🧠 Анализ контактов",
    ):
        result = await process_one(item)
        results.append(result)
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, TypeVar

from aiogram.types import Message

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def progress_tracker(
    message: Message,
    total: int,
    iterable,
    item_name_fn=None,
    prefix: str = "🔄",
) -> AsyncIterator[T]:
    """Iterate over *iterable* and update *message* with progress.

    Edits the Telegram message after each yielded item to show the current
    item name (if available), percentage, and count — e.g.::

        🧠 Анализ контактов: Настя 40% | 2/5

    Handles both sync and async iterables.  Failures while editing the
    message are logged but silently ignored — they never interrupt the
    caller's processing loop.

    Args:
        message:  aiogram ``Message`` whose text will be edited in-place.
        total:    Expected item count (displayed as the denominator).
        iterable: A sync or async iterable of items to yield.
        item_name_fn:
            Optional callable ``item → str`` — the displayed per-item name.
            When omitted (or returns a falsy value) the name portion is
            skipped.
        prefix:   Static text / emoji shown at the front of the message.

    Yields:
        Each item from *iterable*.
    """
    if total <= 0 or iterable is None:
        return

    has_async = hasattr(iterable, "__aiter__")
    if has_async:
        iterator = iterable.__aiter__()
    else:
        iterator = iter(iterable)

    for idx in range(total):
        # ── fetch next item ──────────────────────────────────────────
        try:
            if has_async:
                item = await iterator.__anext__()
            else:
                item = next(iterator)
        except (StopIteration, StopAsyncIteration):
            break

        # ── build progress text ──────────────────────────────────────
        name = item_name_fn(item) if item_name_fn else ""
        pct = int((idx + 1) / total * 100)

        if name:
            text = f"{prefix}: {name} {pct}% | {idx + 1}/{total}"
        else:
            text = f"{prefix} {pct}% | {idx + 1}/{total}"

        # ── edit message (best-effort) ───────────────────────────────
        try:
            await message.edit_text(text)
        except Exception:
            logger.debug("Progress message edit failed", exc_info=True)

        yield item
