"""Document context provider — searches indexed documents via Qdrant.

Implements ContextProvider protocol, pluggable into ContextEngine.
"""

from __future__ import annotations

from src.core.context.spec import ContextChunk
from src.core.rag.document_store import get_document_store


class DocumentProvider:
    """Semantic search provider for user's indexed documents."""

    name = "documents"

    async def get_context(
        self,
        query: str,
        *,
        telegram_id: int,
        contact_id: int | None = None,
        limit: int = 5,
    ) -> list[ContextChunk]:
        """Search indexed documents for content relevant to the query.

        Requires an embedding provider to generate the query vector.
        Falls back gracefully if no provider is available.
        """
        import asyncio

        from src.config import settings
        from src.core.actions.embedding_cache import get as cache_get
        from src.core.actions.embedding_cache import set as cache_set
        from src.llm.router import build_provider
        from src.llm.base import TaskType
        from src.db.session import get_session
        from src.db.repo import get_or_create_user

        # 1. Get embedding for the query
        provider = None
        try:
            async with get_session() as session:
                user = await get_or_create_user(session, telegram_id)
                provider = await build_provider(
                    session, user, task_type=TaskType.BACKGROUND
                )
        except Exception:
            provider = None

        if provider is None:
            return []

        model = getattr(provider, "_embed_model", None) or "text-embedding-3-small"

        # Try cache first
        query_vec = cache_get(query, model)
        if query_vec is None:
            try:
                query_vec = await provider.embed(query[:500])
                cache_set(query[:500], query_vec, model)
            except Exception:
                return []

        if not query_vec:
            return []

        # 2. Search Qdrant documents collection
        store = get_document_store()
        hits = await store.search(
            user_id=telegram_id,
            query_embedding=query_vec,
            limit=limit,
        )

        # 3. Convert to ContextChunk
        return [
            ContextChunk(
                text=(
                    f"[{h['filename']} ч.{h['chunk_index'] + 1}/{h['total_chunks']}]: "
                    f"{h['text'][:500]}"
                ),
                source="documents",
                score=h.get("score", 0.5),
                reason="semantic_doc_match",
            )
            for h in hits
        ]
