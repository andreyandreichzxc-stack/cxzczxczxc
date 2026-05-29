"""Qdrant-backed document store for RAG pipeline.

Manages the ``documents`` Qdrant collection — stores chunked, embedded
document fragments with metadata for semantic search.
"""

from __future__ import annotations

import hashlib as _hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DocumentRecord:
    """Metadata for one chunk in the document store."""

    point_id: int
    user_id: int
    filename: str
    file_path: str
    file_type: str
    chunk_index: int
    total_chunks: int
    text: str
    content_hash: str  # SHA-256 of the original file
    date_indexed: str  # ISO 8601


class DocumentStore:
    """Manages the ``documents`` Qdrant collection."""

    COLLECTION_NAME = "documents"

    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        """Lazy init Qdrant client."""
        if self._client is not None:
            return self._client

        from src.core.actions.vector_store import get_vector_store

        vs = get_vector_store()
        self._client = vs._client  # reuse existing Qdrant client
        return self._client

    def _ensure_collection(self, dim: int = 1536) -> None:
        """Create the documents collection if it doesn't exist."""
        from qdrant_client.models import (
            Distance,
            VectorParams,
        )

        client = self._get_client()
        collections = [c.name for c in client.get_collections().collections]
        if self.COLLECTION_NAME in collections:
            return

        client.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        logger.info(
            "Created Qdrant collection '%s' with dim=%d",
            self.COLLECTION_NAME,
            dim,
        )

    async def upsert_chunks(
        self,
        user_id: int,
        filename: str,
        file_path: str,
        file_type: str,
        chunks: list[str],
        embeddings: list[list[float]],
        content_hash: str,
    ) -> list[DocumentRecord]:
        """Store document chunks with embeddings.

        Args:
            user_id: Owner telegram ID.
            filename: Original filename.
            file_path: Absolute path to the original file.
            file_type: File extension (".pdf", ".md", etc.)
            chunks: Text chunks from the document.
            embeddings: Corresponding embedding vectors.
            content_hash: SHA-256 hash of the original file content.

        Returns:
            List of stored DocumentRecord objects.
        """
        from datetime import datetime, timezone
        from qdrant_client.models import PointStruct

        dim = len(embeddings[0]) if embeddings else 1536
        self._ensure_collection(dim)

        client = self._get_client()
        now = datetime.now(timezone.utc).isoformat()
        total = len(chunks)
        records: list[DocumentRecord] = []
        points: list[PointStruct] = []

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            # Stable hash across process restarts (Python hash() is randomized)
            point_id = int(
                _hashlib.md5(f"{content_hash}:{i}".encode()).hexdigest(), 16
            ) % (10**9)
            payload = {
                "user_id": user_id,
                "filename": filename,
                "file_path": file_path,
                "file_type": file_type,
                "chunk_index": i,
                "total_chunks": total,
                "text": chunk[:4000],
                "content_hash": content_hash,
                "date_indexed": now,
            }
            records.append(DocumentRecord(point_id=point_id, **payload))
            points.append(
                PointStruct(
                    id=point_id,
                    vector=emb,
                    payload=payload,
                )
            )

        client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=points,
        )
        logger.info(
            "Upserted %d chunks for '%s' (hash=%s)",
            total,
            filename,
            content_hash[:12],
        )
        return records

    async def delete_document(self, content_hash: str) -> int:
        """Delete all chunks belonging to a document by content hash.

        Returns the number of points deleted.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        client = self._get_client()
        result = client.delete(
            collection_name=self.COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="content_hash",
                        match=MatchValue(value=content_hash),
                    )
                ]
            ),
        )
        deleted = result.status == "completed"
        logger.info("Deleted document %s: %s", content_hash[:12], deleted)
        return 1 if deleted else 0

    async def search(
        self,
        user_id: int,
        query_embedding: list[float],
        limit: int = 5,
        score_threshold: float = 0.4,
    ) -> list[dict]:
        """Search for document chunks semantically similar to the query.

        Returns list of dicts: {text, filename, file_type, chunk_index, total_chunks, score}
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        client = self._get_client()
        try:
            results = client.search(
                collection_name=self.COLLECTION_NAME,
                query_vector=query_embedding,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_id),
                        )
                    ]
                ),
                limit=limit,
                score_threshold=score_threshold,
            )
        except Exception:
            # Collection might not exist yet
            return []

        return [
            {
                "text": r.payload.get("text", "") if r.payload else "",
                "filename": r.payload.get("filename", "") if r.payload else "",
                "file_type": r.payload.get("file_type", "") if r.payload else "",
                "chunk_index": r.payload.get("chunk_index", 0) if r.payload else 0,
                "total_chunks": r.payload.get("total_chunks", 1) if r.payload else 1,
                "score": r.score,
            }
            for r in results
        ]

    async def list_documents(self, user_id: int) -> list[dict]:
        """Get all indexed document filenames with metadata."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        client = self._get_client()
        try:
            # Use scroll to get all points for the user
            points, _ = client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_id),
                        )
                    ]
                ),
                limit=1000,
            )
        except Exception:
            return []

        seen: dict[str, dict] = {}
        for p in points:
            if not p.payload:
                continue
            fname = p.payload.get("filename", "unknown")
            if fname not in seen:
                seen[fname] = {
                    "filename": fname,
                    "file_type": p.payload.get("file_type", ""),
                    "total_chunks": p.payload.get("total_chunks", 0),
                    "date_indexed": p.payload.get("date_indexed", ""),
                    "content_hash": p.payload.get("content_hash", ""),
                }
        return list(seen.values())

    async def get_document_count(self, user_id: int) -> int:
        """Count total indexed chunks for a user."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        client = self._get_client()
        try:
            result = client.count(
                collection_name=self.COLLECTION_NAME,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_id),
                        )
                    ]
                ),
            )
            return result.count
        except Exception:
            return 0


# Module-level singleton
_store: DocumentStore | None = None


def get_document_store() -> DocumentStore:
    """Get or create the singleton DocumentStore."""
    global _store
    if _store is None:
        _store = DocumentStore()
    return _store
