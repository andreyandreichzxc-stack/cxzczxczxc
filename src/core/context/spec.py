from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class ContextChunk:
    """Один фрагмент контекста."""

    text: str
    source: str  # "memory", "fts", "vector", "wiki_context", "frozen"
    score: float = 0.0  # relevance score
    reason: str = ""  # "pinned", "fresh", "semantic_match", etc.


@runtime_checkable
class ContextProvider(Protocol):
    """Protocol for pluggable context sources."""

    name: str

    async def get_context(
        self,
        query: str,
        *,
        telegram_id: int,
        contact_id: int | None = None,
        limit: int = 8,
    ) -> list[ContextChunk]: ...
