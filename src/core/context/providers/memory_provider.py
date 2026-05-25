"""Memory context provider — SQLite-based facts (pinned, fresh, frequent, self, contact)."""

from __future__ import annotations

from src.core.context.spec import ContextChunk
from src.core.memory.memory_recall import recall as _recall


class MemoryProvider:
    name = "memory"

    async def get_context(self, query, *, telegram_id, contact_id=None, limit=8):
        # Use recall with mode="normal" — that includes:
        # pinned, task-context, fresh, frequent, self-facts, contact-specific
        # but NOT deep memory or semantic search (those are separate providers)
        result = await _recall(
            telegram_id=telegram_id,
            contact_id=contact_id,
            query=query,
            limit=limit,
            mode="normal",
            include_deep=False,
        )
        return [
            ContextChunk(
                text=f.fact,
                source="memory",
                score=f.confidence or 0.5,
                reason=f.reason,
            )
            for f in result.facts
        ]
