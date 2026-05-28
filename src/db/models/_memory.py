"""Memory subsystem models: Memory, MemoryLink, MemoryCluster, MemoryClusterMember, MemoryCandidate."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from ._base import Base


class Memory(Base):
    """Факты о владельце и контактах, извлекаемые из переписок и разговоров с ботом."""

    __tablename__ = "memories"
    __table_args__ = (
        Index("ix_mem_active_contact", "is_active", "contact_id"),
        Index("ix_mem_user_active", "user_id", "is_active"),
        Index("ix_memories_user_type_active", "user_id", "memory_type", "is_active"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    contact_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, index=True
    )
    fact: Mapped[str] = mapped_column(Text)
    sentiment: Mapped[str | None] = mapped_column(
        String(16), nullable=True, index=True
    )  # positive, negative, neutral
    source: Mapped[str] = mapped_column(String(16), default="chat")  # chat, user, auto
    confidence: Mapped[float] = mapped_column(
        Float, default=0.5
    )  # 0.0–1.0 уверенность в факте
    times_mentioned: Mapped[int] = mapped_column(
        Integer, default=1
    )  # сколько раз подтверждён
    message_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True
    )  # исходное сообщение
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True, index=True
    )  # активен / опровергнут
    cluster_topic: Mapped[str | None] = mapped_column(
        String(128), nullable=True
    )  # тема-кластер
    embedding_hash: Mapped[str | None] = mapped_column(
        String(16), nullable=True, index=True
    )  # хеш для дедупликации
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    validity_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    validity_end: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    importance: Mapped[float] = mapped_column(Float, default=0.5)  # 0.0–1.0
    decay_rate: Mapped[float] = mapped_column(Float, default=0.07)  # скорость забывания
    memory_tier: Mapped[int] = mapped_column(
        Integer, default=1
    )  # 1=эпизод, 2=недельное, 3=месячное
    temporal_layer: Mapped[str | None] = mapped_column(
        String(16), nullable=True, index=True
    )
    # null | "recent" (≤7d) | "medium" (8-30d) | "longterm" (>30d)

    tags: Mapped[str | None] = mapped_column(
        String(256), nullable=True
    )  # comma-separated: "работа,деньги"

    memory_type: Mapped[str | None] = mapped_column(
        String(24), nullable=True, index=True
    )
    # personal | contact_fact | relationship | task | preference | temporary

    use_count: Mapped[int] = mapped_column(Integer, default=0)
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    pinned: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    related_memory_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, index=True
    )  # ссылка на другой Memory.id
    relation_type: Mapped[str | None] = mapped_column(
        String(16), nullable=True
    )  # cause, effect, contradicts, supports, continues, example_of


class MemoryLink(Base):
    """Many-to-many связи между фактами памяти с весами."""

    __tablename__ = "memory_links"
    __table_args__ = (
        Index("ix_ml_source", "source_id"),
        Index("ix_ml_target", "target_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    source_id: Mapped[int] = mapped_column(
        ForeignKey("memories.id", ondelete="CASCADE"), index=True
    )
    target_id: Mapped[int] = mapped_column(
        ForeignKey("memories.id", ondelete="CASCADE"), index=True
    )
    weight: Mapped[float] = mapped_column(Float, default=0.5)  # 0.0-1.0 сила связи
    relation_type: Mapped[str | None] = mapped_column(
        String(32), nullable=True
    )  # cause/effect/contradicts/supports/continues/co_temporal/co_entity/preceded
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class MemoryCluster(Base):
    """Группа связанных фактов по теме."""

    __tablename__ = "memory_clusters"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    topic: Mapped[str] = mapped_column(String(128))
    summary: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # LLM-саммари кластера
    fact_count: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class MemoryClusterMember(Base):
    """Связь many-to-many: факт → кластер."""

    __tablename__ = "memory_cluster_members"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    memory_id: Mapped[int] = mapped_column(
        ForeignKey("memories.id", ondelete="CASCADE"), index=True
    )
    cluster_id: Mapped[int] = mapped_column(
        ForeignKey("memory_clusters.id", ondelete="CASCADE"), index=True
    )
    relevance_score: Mapped[float] = mapped_column(Float, default=0.5)
    added_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class MemoryCandidate(Base):
    """Факты на подтверждение — черновик памяти."""

    __tablename__ = "memory_candidates"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    contact_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    fact: Mapped[str] = mapped_column(Text)
    sentiment: Mapped[str | None] = mapped_column(String(16), nullable=True)
    memory_type: Mapped[str | None] = mapped_column(String(24), nullable=True)
    source: Mapped[str] = mapped_column(String(16), default="chat")
    importance: Mapped[float] = mapped_column(Float, default=0.5)
    decay_rate: Mapped[float] = mapped_column(Float, default=0.07)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
