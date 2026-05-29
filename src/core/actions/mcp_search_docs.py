"""MCP tool for searching indexed documents (RAG pipeline).

Registered as ``search_docs`` — finds relevant chunks in user's document store.
"""

from __future__ import annotations

import logging
from typing import Any

from src.core.actions.tool_registry import tool

logger = logging.getLogger(__name__)


@tool(
    name="search_docs",
    description=(
        "Семантический поиск по проиндексированным документам пользователя "
        "(RAG). Ищет релевантные фрагменты в PDF, DOCX, TXT, MD, HTML "
        "файлах, которые были проиндексированы в data/documents/. "
        "Возвращает цитаты из документов."
    ),
    category="knowledge",
    risk="low",
    params={
        "query": "строка — что ищем (на русском или английском)",
        "limit": "int (1-10) — сколько результатов (по умолчанию 5)",
    },
)
async def search_docs(
    query: str = "",
    limit: int = 5,
    **kwargs: Any,
) -> dict[str, Any]:
    """Поиск по проиндексированным документам."""
    if not query or not query.strip():
        return {"error": "Укажи что искать (параметр query)"}

    limit = max(1, min(10, limit))

    try:
        # Get user
        user = kwargs.get("user")
        if user is not None:
            telegram_id = user.telegram_id
        else:
            from src.config import settings

            telegram_id = settings.owner_telegram_id

        if not telegram_id:
            return {"error": "Не удалось определить пользователя"}

        # Get embedding
        provider = kwargs.get("provider")
        if provider is None:
            return {"error": "Недоступен embedding-провайдер"}

        from src.core.actions.embedding_cache import get as cache_get
        from src.core.actions.embedding_cache import set as cache_set

        model = getattr(provider, "_embed_model", None) or "text-embedding-3-small"
        query_vec = cache_get(query, model)
        if query_vec is None:
            query_vec = await provider.embed(query[:500])
            cache_set(query[:500], query_vec, model)

        # Search Qdrant
        from src.core.rag.document_store import get_document_store

        store = get_document_store()
        hits = await store.search(
            user_id=telegram_id,
            query_embedding=query_vec,
            limit=limit,
        )

        if not hits:
            return {
                "ok": True,
                "query": query,
                "results": [],
                "message": (
                    "Ничего не найдено в документах. Добавь файлы в data/documents/."
                ),
            }

        return {
            "ok": True,
            "query": query,
            "total": len(hits),
            "results": [
                {
                    "filename": h["filename"],
                    "chunk": f"{h['chunk_index'] + 1}/{h['total_chunks']}",
                    "text": h["text"][:600],
                    "score": round(h.get("score", 0.5), 3),
                }
                for h in hits
            ],
        }

    except Exception as e:
        logger.debug("search_docs failed: %s", e)
        return {"error": str(e)[:300]}
