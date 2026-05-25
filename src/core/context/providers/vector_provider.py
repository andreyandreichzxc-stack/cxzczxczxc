"""Vector context provider — Qdrant semantic search.

Note: recall() does not yet support mode="vector_only".
When it does, this provider will return ONLY semantic/vector results (step 3).
For now it uses mode="deep" (include_deep=False) which includes
all non-deep steps including semantic/hybrid search.
"""

from __future__ import annotations

from src.core.context.spec import ContextChunk


class VectorProvider:
    name = "vector"

    async def get_context(self, query, *, telegram_id, contact_id=None, limit=8):
        from src.core.memory.memory_recall import recall as _recall

        # TODO: switch to mode="vector_only" when recall() supports it
        result = await _recall(
            telegram_id=telegram_id,
            contact_id=contact_id,
            query=query,
            limit=limit,
            mode="deep",
            include_deep=False,
        )
        return [
            ContextChunk(
                text=f.fact,
                source="vector",
                score=f.confidence or 0.5,
                reason="semantic_match",
            )
            for f in result.facts
        ]
