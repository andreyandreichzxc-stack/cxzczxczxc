"""Qdrant embedded в data/qdrant. Коллекция messages пересоздаётся при первом upsert
с актуальным размером embedding'а — это позволяет менять провайдера без миграций."""

import asyncio
import hashlib
import logging
from dataclasses import dataclass

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

    async def _ensure_collection(self, dim: int) -> None:
        if self._dim == dim:
            return
        async with self._lock:
            if self._dim == dim:
                return

            def _check_or_create() -> None:
                existing = {c.name for c in self._client.get_collections().collections}
                if COLLECTION in existing:
                    info = self._client.get_collection(COLLECTION)
                    actual = info.config.params.vectors.size
                    if actual != dim:
                        logger.warning(
                            "Recreating Qdrant collection %s: dim %d → %d",
                            COLLECTION,
                            actual,
                            dim,
                        )
                        self._client.delete_collection(COLLECTION)
                        self._client.create_collection(
                            COLLECTION,
                            vectors_config=qmodels.VectorParams(
                                size=dim, distance=qmodels.Distance.COSINE
                            ),
                        )
                else:
                    self._client.create_collection(
                        COLLECTION,
                        vectors_config=qmodels.VectorParams(
                            size=dim, distance=qmodels.Distance.COSINE
                        ),
                    )

            await asyncio.to_thread(_check_or_create)
            self._dim = dim

    async def _ensure_memory_collection(self, dim: int) -> None:
        if self._memory_dim == dim:
            return
        async with self._lock:
            if self._memory_dim == dim:
                return

            def _check_or_create() -> None:
                existing = {c.name for c in self._client.get_collections().collections}
                if MEMORY_COLLECTION in existing:
                    info = self._client.get_collection(MEMORY_COLLECTION)
                    actual = info.config.params.vectors.size
                    if actual != dim:
                        logger.warning(
                            "Recreating Qdrant collection %s: dim %d → %d",
                            MEMORY_COLLECTION,
                            actual,
                            dim,
                        )
                        self._client.delete_collection(MEMORY_COLLECTION)
                        self._client.create_collection(
                            MEMORY_COLLECTION,
                            vectors_config=qmodels.VectorParams(
                                size=dim, distance=qmodels.Distance.COSINE
                            ),
                        )
                else:
                    self._client.create_collection(
                        MEMORY_COLLECTION,
                        vectors_config=qmodels.VectorParams(
                            size=dim, distance=qmodels.Distance.COSINE
                        ),
                    )

            await asyncio.to_thread(_check_or_create)
            self._memory_dim = dim

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

    async def check_health_and_recover(self) -> bool:
        """Проверяет целостность Qdrant и восстанавливает при повреждении.
        Возвращает True если здоров, False если восстановился."""
        try:
            self._client.get_collections()
            return True
        except Exception:
            logger.exception("Qdrant health check failed — attempting recovery")
            try:
                import shutil

                qdrant_dir = settings.data_dir / "qdrant"
                self._client.close()
                shutil.rmtree(str(qdrant_dir), ignore_errors=True)
                qdrant_dir.mkdir(parents=True, exist_ok=True)
                self._client = QdrantClient(path=str(qdrant_dir))
                await self._ensure_collection(1536)
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


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
