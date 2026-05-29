"""Messaging models: Message, AutoReplyLog, TranscriptionCache, PendingAction,
Notification, NewsTopic, Commitment, IndexJob, Folder."""

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
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ._base import Base, User


class Message(Base):
    """Кэш сообщений из чатов."""

    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint(
            "user_id", "peer_id", "message_id", name="uq_msg_user_peer_id"
        ),
        Index("ix_messages_user_peer_date", "user_id", "peer_id", "date"),
        Index("ix_messages_peer_user_date", "peer_id", "user_id", "date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    peer_id: Mapped[int] = mapped_column(BigInteger, index=True)
    message_id: Mapped[int] = mapped_column(BigInteger)
    sender_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    sender_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    is_outgoing: Mapped[bool] = mapped_column(Boolean, default=False)
    date: Mapped[datetime] = mapped_column(DateTime, index=True)
    kind: Mapped[str] = mapped_column(
        String(16), default="text"
    )  # text | voice | audio | document | photo | other
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    media_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    extracted_text: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # для документов
    indexed_in_vector: Mapped[bool] = mapped_column(Boolean, default=False)


class Commitment(Base):
    """Извлечённые обещания."""

    __tablename__ = "commitments"
    __table_args__ = (Index("ix_commitments_user_status", "user_id", "status"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    peer_id: Mapped[int] = mapped_column(BigInteger, index=True)
    peer_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    message_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    source_memory_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, index=True
    )
    direction: Mapped[str] = mapped_column(String(8))  # mine | theirs
    text: Mapped[str] = mapped_column(Text)
    deadline_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    status: Mapped[str] = mapped_column(
        String(16), default="open"
    )  # open | done | cancelled | reminded
    last_reminded_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class AutoReplyLog(Base):
    """Лог авто-ответов для прозрачности."""

    __tablename__ = "auto_reply_logs"
    __table_args__ = (
        Index("ix_auto_reply_logs_cooldown", "user_id", "peer_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    peer_id: Mapped[int] = mapped_column(BigInteger, index=True)
    peer_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    incoming_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    reply_text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), index=True
    )


class IndexJob(Base):
    """Состояние индексации чата (последний обработанный message_id)."""

    __tablename__ = "index_jobs"
    __table_args__ = (
        UniqueConstraint("user_id", "peer_id", name="uq_index_user_peer"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    peer_id: Mapped[int] = mapped_column(BigInteger, index=True)
    last_indexed_message_id: Mapped[int] = mapped_column(BigInteger, default=0)
    last_indexed_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class TranscriptionCache(Base):
    """Кэш транскрипций по telegram media file_id."""

    __tablename__ = "transcription_cache"

    file_id: Mapped[str] = mapped_column(String(256), primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class PendingAction(Base):
    """Промежуточные действия, ожидающие подтверждения (отправка сообщения и т.п.)."""

    __tablename__ = "pending_actions"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(32))  # send_message | catchup_reply | ...
    payload: Mapped[str] = mapped_column(Text)  # JSON
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class NewsTopic(Base):
    """Темы-фавориты для авто-новостей. Каждая утром собирается отдельным дайджестом."""

    __tablename__ = "news_topics"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    topic: Mapped[str] = mapped_column(String(256))
    hours: Mapped[int] = mapped_column(Integer, default=24)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class Notification(Base):
    """Очередь уведомлений с группировкой по теме."""

    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    topic: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True
    )  # memory, conflict, habit, digest, follow_up, ...
    priority: Mapped[int] = mapped_column(
        Integer, nullable=False, default=2
    )  # 0=CRITICAL, 1=HIGH, 2=MEDIUM, 3=LOW
    category: Mapped[str | None] = mapped_column(
        String(128), nullable=True
    )  # подкатегория для группировки
    text: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    batch_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    flushed_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, index=True
    )

    # Приоритеты как константы
    PRIORITY_CRITICAL = 0
    PRIORITY_HIGH = 1
    PRIORITY_MEDIUM = 2
    PRIORITY_LOW = 3


class Folder(Base):
    """Папка (кастомный список чатов) пользователя Telegram."""

    __tablename__ = "folders"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    telegram_folder_id: Mapped[int] = mapped_column(Integer, index=True)
    title: Mapped[str] = mapped_column(String(128))
    emoji: Mapped[str | None] = mapped_column(String(8), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class ConversationSummary(Base):
    """Сжатые сводки диалогов — persist between restarts."""

    __tablename__ = "conversation_summaries"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    last_peer_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    last_peer_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    summary_text: Mapped[str] = mapped_column(Text)
    turn_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # Relationship to User
    user: Mapped["User"] = relationship("User", back_populates="conversation_summaries")


class ScheduledMessage(Base):
    """Отложенное сообщение — будет отправлено в указанное время."""

    __tablename__ = "scheduled_messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    contact_name: Mapped[str] = mapped_column(
        String(255), nullable=False, comment="Имя контакта/чата кому отправить"
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    send_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, comment="UTC когда отправить"
    )
    status: Mapped[str] = mapped_column(
        String(16), default="pending", comment="pending | sent | failed | cancelled"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    sent_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    via_userbot: Mapped[bool] = mapped_column(
        Boolean, default=True, comment="Отправить через userbot"
    )

    user: Mapped["User"] = relationship("User", back_populates="scheduled_messages")
