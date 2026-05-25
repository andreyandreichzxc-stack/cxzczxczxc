"""Frozen snapshot provider — cached top-3 facts from last recall."""

from __future__ import annotations

from src.core.context.spec import ContextChunk


class FrozenProvider:
    name = "frozen"

    def __init__(self) -> None:
        self._frozen: list[ContextChunk] = []
        self._frozen_telegram_id: int | None = None

    def set_frozen(self, telegram_id: int, chunks: list[dict]) -> None:
        self._frozen_telegram_id = telegram_id
        self._frozen = [
            ContextChunk(text=c["fact"], source="frozen", reason="cached_snapshot")
            for c in chunks
        ]

    async def get_context(self, query, *, telegram_id, contact_id=None, limit=8):
        if telegram_id == self._frozen_telegram_id:
            return self._frozen[:limit]
        return []


# Module-level singleton so maestro can set frozen snapshot
frozen_provider = FrozenProvider()
