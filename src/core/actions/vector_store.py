"""Qdrant embedded в data/qdrant. Коллекции НЕ пересоздаются автоматически при
изменении размерности эмбеддинга — устанавливается флаг reindex_required.
Явный reindex через /index команду (reindex_collection)."""

import asyncio
import hashlib
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.config import settings


logger = logging.getLogger(__name__)


COLLECTION = "messages"
MEMORY_COLLECTION = "memory_facts"


@dataclass
class VectorHit:
    user_id: int
    peer_id: int
    peer_name: str | None
    message_id: int
    text: str
    date_iso: str | None
    score: float


class VectorStore:
    def __init__(self) -> None:
        path = settings.data_dir / "qdrant"
        path.mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(path))
        self._lock = asyncio.Lock()
        self._dim: int | None = None
        self._memory_dim: int | None = None
        self._reindex_required: bool = False
        self._memory_reindex_required: bool = False
        self._index_status: str = "unknown"
        self._indexed_at: str | None = None
        self.embedding_provider: str | None = None
        self.embedding_model: str | None = None

    async def _ensure_collection(self, dim: int) -> None:
        if self._dim == dim:
            self._reindex_required = False
            return
        async with self._lock:
            if self._dim == dim:
                self._reindex_required = False
                return

            def _check_or_create() -> bool:
                """Returns True if ready, False if reindex is required."""
                existing = {c.name for c in self._client.get_collections().collections}
                if COLLECTION in existing:
                    info = self._client.get_collection(COLLECTION)
                    actual = info.config.params.vectors.size
                    if actual != dim:
                        logger.warning(
                            "Dimension mismatch for %s: has dim %d, "
                            "requested dim %d — reindex required. "
                            "Data NOT deleted. "
                            "Call reindex_collection(%d) explicitly.",
                            COLLECTION,
                            actual,
                            dim,
                            dim,
                        )
                        return False
                else:
                    self._client.create_collection(
                        COLLECTION,
                        vectors_config=qmodels.VectorParams(
                            size=dim, distance=qmodels.Distance.COSINE
                        ),
                    )
                return True

            ready = await asyncio.to_thread(_check_or_create)
            if ready:
                self._dim = dim
                self._reindex_required = False
                self._index_status = "ready"
            else:
                self._reindex_required = True
                self._index_status = "reindex_required"

    async def _ensure_memory_collection(self, dim: int) -> None:
        if self._memory_dim == dim:
            self._memory_reindex_required = False
            return
        async with self._lock:
            if self._memory_dim == dim:
                self._memory_reindex_required = False
                return

            def _check_or_create() -> bool:
                """Returns True if ready, False if reindex is required."""
                existing = {c.name for c in self._client.get_collections().collections}
                if MEMORY_COLLECTION in existing:
                    info = self._client.get_collection(MEMORY_COLLECTION)
                    actual = info.config.params.vectors.size
                    if actual != dim:
                        logger.warning(
                            "Dimension mismatch for %s: has dim %d, "
                            "requested dim %d — reindex required. "
                            "Data NOT deleted. "
                            "Call reindex_memory_collection(%d) explicitly.",
                            MEMORY_COLLECTION,
                            actual,
                            dim,
                            dim,
                        )
                        return False
                else:
                    self._client.create_collection(
                        MEMORY_COLLECTION,
                        vectors_config=qmodels.VectorParams(
                            size=dim, distance=qmodels.Distance.COSINE
                        ),
                    )
                return True

            ready = await asyncio.to_thread(_check_or_create)
            if ready:
                self._memory_dim = dim
                self._memory_reindex_required = False
                self._index_status = "ready"
            else:
                self._memory_reindex_required = True
                self._index_status = "reindex_required"

    async def upsert_memory(
        self,
        *,
        memory_id: int,
        user_id: int,
        contact_id: int | None,
        fact: str,
        embedding: list[float],
        importance: float = 0.5,
        confidence: float = 0.5,
        created_at: str | None = None,
    ) -> None:
        """Сохраняет эмбеддинг факта памяти в коллекцию memory_facts."""
        await self._ensure_memory_collection(len(embedding))
        if self._memory_reindex_required:
            logger.warning(
                "Skipping memory upsert — %s has mismatched dimensions, "
                "call reindex_memory_collection(%d) first",
                MEMORY_COLLECTION,
                len(embedding),
            )
            return

        def _do() -> None:
            self._client.upsert(
                collection_name=MEMORY_COLLECTION,
                points=[
                    qmodels.PointStruct(
                        id=memory_id,
                        vector=embedding,
                        payload={
                            "user_id": user_id,
                            "contact_id": contact_id,
                            "fact": fact,
                            "memory_id": memory_id,
                            "importance": importance,
                            "confidence": confidence,
                            "created_at": created_at,
                            "embedding": embedding,  # store vector for cosine similarity
                        },
                    )
                ],
            )

        await asyncio.to_thread(_do)

    async def search_similar_memories(
        self,
        *,
        user_id: int,
        embedding: list[float],
        threshold: float = 0.85,
        limit: int = 5,
        contact_id: int | None = None,
    ) -> list[dict]:
        """Поиск похожих фактов в коллекции memory_facts по cosine similarity.

        Если contact_id передан — возвращаются только факты о контакте или общие
        (contact_id == null).
        """
        await self._ensure_memory_collection(len(embedding))
        if self._memory_dim is None:
            existing = {c.name for c in self._client.get_collections().collections}
            if MEMORY_COLLECTION in existing:
                info = self._client.get_collection(MEMORY_COLLECTION)
                self._memory_dim = info.config.params.vectors.size
            else:
                return []
        if len(embedding) != self._memory_dim:
            logger.warning(
                "Memory embedding dim %d != collection dim %d — re-index needed?",
                len(embedding),
                self._memory_dim,
            )
            return []

        flt = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="user_id", match=qmodels.MatchValue(value=user_id)
                )
            ]
        )
        if contact_id is not None:
            flt.should = [
                qmodels.FieldCondition(
                    key="contact_id", match=qmodels.MatchValue(value=contact_id)
                ),
                qmodels.FieldCondition(key="contact_id", is_null=True),
            ]

        def _do() -> list[qmodels.ScoredPoint]:
            response = self._client.query_points(
                collection_name=MEMORY_COLLECTION,
                query=embedding,
                limit=limit,
                query_filter=flt,
                score_threshold=threshold,
            )
            return response.points

        raw = await asyncio.to_thread(_do)
        return [
            {
                "memory_id": p.payload.get("memory_id"),
                "fact": p.payload.get("fact", ""),
                "score": float(p.score),
                "contact_id": p.payload.get("contact_id"),
                "importance": p.payload.get("importance", 0.5),
                "confidence": p.payload.get("confidence", 0.5),
                "created_at": p.payload.get("created_at"),
                "embedding": p.payload.get("embedding"),  # for cosine similarity
            }
            for p in raw
        ]

    @staticmethod
    def _point_id(user_id: int, peer_id: int, message_id: int) -> int:
        return int(
            hashlib.md5(f"{user_id}:{peer_id}:{message_id}".encode()).hexdigest()[:16],
            16,
        )

    async def upsert(
        self,
        *,
        user_id: int,
        peer_id: int,
        peer_name: str | None,
        message_id: int,
        text: str,
        date_iso: str | None,
        embedding: list[float],
    ) -> None:
        await self._ensure_collection(len(embedding))
        if self._reindex_required:
            logger.warning(
                "Skipping upsert to %s — mismatched dimensions, "
                "call reindex_collection(%d) first",
                COLLECTION,
                len(embedding),
            )
            return

        def _do() -> None:
            self._client.upsert(
                collection_name=COLLECTION,
                points=[
                    qmodels.PointStruct(
                        id=self._point_id(user_id, peer_id, message_id),
                        vector=embedding,
                        payload={
                            "user_id": user_id,
                            "peer_id": peer_id,
                            "peer_name": peer_name,
                            "message_id": message_id,
                            "text": text,
                            "date_iso": date_iso,
                        },
                    )
                ],
            )

        await asyncio.to_thread(_do)

    async def upsert_batch(
        self,
        *,
        points: list[dict],
    ) -> None:
        """Batch upsert many points into Qdrant in a single call.

        Each dict must contain: user_id, peer_id, peer_name, message_id,
        text, date_iso, embedding.
        """
        if not points:
            return
        first = points[0]
        dim = len(first["embedding"])
        await self._ensure_collection(dim)
        if self._reindex_required:
            logger.warning("Skipping batch upsert — reindex required")
            return

        qdrant_points = [
            qmodels.PointStruct(
                id=self._point_id(p["user_id"], p["peer_id"], p["message_id"]),
                vector=p["embedding"],
                payload={
                    k: p[k]
                    for k in (
                        "user_id",
                        "peer_id",
                        "peer_name",
                        "message_id",
                        "text",
                        "date_iso",
                    )
                    if k in p
                },
            )
            for p in points
        ]

        def _do() -> None:
            self._client.upsert(collection_name=COLLECTION, points=qdrant_points)

        await asyncio.to_thread(_do)

    async def search(
        self,
        *,
        user_id: int,
        embedding: list[float],
        limit: int = 10,
        peer_id: int | None = None,
    ) -> list[VectorHit]:
        if self._dim is None:
            existing = {c.name for c in self._client.get_collections().collections}
            if COLLECTION in existing:
                info = self._client.get_collection(COLLECTION)
                self._dim = info.config.params.vectors.size
            else:
                return []
        if len(embedding) != self._dim:
            logger.warning(
                "Embedding dim %d != collection dim %d — re-index needed?",
                len(embedding),
                self._dim,
            )
            return []
        flt = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="user_id", match=qmodels.MatchValue(value=user_id)
                )
            ]
        )
        if peer_id is not None:
            flt.must.append(
                qmodels.FieldCondition(
                    key="peer_id", match=qmodels.MatchValue(value=peer_id)
                )
            )

        def _do() -> list[qmodels.ScoredPoint]:
            response = self._client.query_points(
                collection_name=COLLECTION,
                query=embedding,
                limit=limit,
                query_filter=flt,
            )
            return response.points

        raw = await asyncio.to_thread(_do)
        return [
            VectorHit(
                user_id=p.payload.get("user_id"),
                peer_id=p.payload.get("peer_id"),
                peer_name=p.payload.get("peer_name"),
                message_id=p.payload.get("message_id"),
                text=p.payload.get("text", ""),
                date_iso=p.payload.get("date_iso"),
                score=float(p.score),
            )
            for p in raw
        ]

    async def reindex_collection(
        self, dim: int, *, provider: str = "", model: str = ""
    ) -> None:
        """Explicitly drop and recreate the 'messages' collection.
        Use ONLY from /index command — this DELETES all existing vectors.
        """
        async with self._lock:

            def _recreate() -> None:
                self._client.delete_collection(COLLECTION)
                self._client.create_collection(
                    COLLECTION,
                    vectors_config=qmodels.VectorParams(
                        size=dim, distance=qmodels.Distance.COSINE
                    ),
                )

            await asyncio.to_thread(_recreate)
            self._dim = dim
            self._reindex_required = False
            self._index_status = "ready"
            self._indexed_at = datetime.now(timezone.utc).isoformat()
            if provider:
                self.embedding_provider = provider
            if model:
                self.embedding_model = model
            logger.warning(
                "%s recreated with dim %d — re-indexing needed",
                COLLECTION,
                dim,
            )

    async def reindex_memory_collection(
        self, dim: int, *, provider: str = "", model: str = ""
    ) -> None:
        """Explicitly drop and recreate the 'memory_facts' collection.
        Use ONLY from /index command — this DELETES all existing vectors.
        """
        async with self._lock:

            def _recreate() -> None:
                self._client.delete_collection(MEMORY_COLLECTION)
                self._client.create_collection(
                    MEMORY_COLLECTION,
                    vectors_config=qmodels.VectorParams(
                        size=dim, distance=qmodels.Distance.COSINE
                    ),
                )

            await asyncio.to_thread(_recreate)
            self._memory_dim = dim
            self._memory_reindex_required = False
            self._index_status = "ready"
            self._indexed_at = datetime.now(timezone.utc).isoformat()
            if provider:
                self.embedding_provider = provider
            if model:
                self.embedding_model = model
            logger.warning(
                "%s recreated with dim %d — re-indexing needed",
                MEMORY_COLLECTION,
                dim,
            )

    async def check_health_and_recover(self) -> bool:
        """Проверяет целостность Qdrant и восстанавливается при повреждении.
        Возвращает True если здоров, False если восстановился.

        WARNING: Recovery destroys ALL vector data. Only triggered for
        persistent corruption (not transient failures).
        """
        try:
            self._client.get_collections()
            return True
        except Exception:
            logger.exception("Qdrant health check failed")

            # Try a simple reconnect first (transient failure?)
            try:
                qdrant_dir = settings.data_dir / "qdrant"
                self._client.close()
                self._client = QdrantClient(path=str(qdrant_dir))
                self._client.get_collections()
                logger.info("Qdrant reconnected successfully")
                return True
            except Exception:
                logger.error("Qdrant reconnect failed — storage may be corrupted")

            # CORRUPTION: only recovery path
            # Notify owner before wiping
            try:
                from src.core.scheduling.notification_queue import notification_queue

                await notification_queue.enqueue(
                    topic="system",
                    text=(
                        "⚠️ Qdrant повреждён, векторный индекс сброшен. "
                        "Семантический поиск временно недоступен. "
                        "Запусти /index для восстановления."
                    ),
                    priority=1,
                )
            except Exception:
                pass

            try:
                import shutil

                qdrant_dir = settings.data_dir / "qdrant"
                self._client.close()
                shutil.rmtree(str(qdrant_dir), ignore_errors=True)
                qdrant_dir.mkdir(parents=True, exist_ok=True)
                self._client = QdrantClient(path=str(qdrant_dir))
                known_dim = self._dim or settings.embedding_dim
                await self._ensure_collection(known_dim)
                logger.warning("Qdrant recovered — old data lost, re-index needed")
                from src.core.scheduling.notification_queue import notification_queue
                from src.db.models import Notification

                try:
                    await notification_queue.enqueue(
                        topic="qdrant_health",
                        text="⚠️ Qdrant был повреждён и восстановлен. Нужен /index для переиндексации.",
                        priority=Notification.PRIORITY_HIGH,
                    )
                except Exception:
                    pass
                return False
            except Exception:
                logger.exception("Qdrant recovery failed")
                return False

    async def shutdown(self) -> None:
        """Graceful shutdown — закрывает Qdrant клиент."""
        try:
            if self._client:
                self._client.close()
        except Exception:
            logger.exception("vector_store shutdown failed")


_vector_store: VectorStore | None = None
_vector_store_lock = threading.Lock()


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        with _vector_store_lock:
            if _vector_store is None:  # double-checked locking
                _vector_store = VectorStore()
    return _vector_store
