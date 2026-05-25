"""LLM-WIKI context provider — searches data/contexts/*.md files."""

from __future__ import annotations

import asyncio

from src.core.context.spec import ContextChunk
from src.core.memory.context_files import search_in_contexts


class WikiContextProvider:
    name = "wiki_context"

    async def get_context(self, query, *, telegram_id, contact_id=None, limit=8):
        results = await asyncio.to_thread(search_in_contexts, query, limit=limit)
        return [
            ContextChunk(
                text=f"[{r['key']}]: {r['snippet']}",
                source="wiki_context",
                score=r.get("rank", 0.5),
                reason="context_file_match",
            )
            for r in results
        ]
