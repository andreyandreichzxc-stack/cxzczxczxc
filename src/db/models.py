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
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    telegram_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    last_seen_online: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    absence_status: Mapped[str | None] = mapped_column(
        String(16), nullable=True
    )  # null | "away" | "soon_back"
    absence_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    global_style_profile: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # JSON
    global_style_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True
    )

    settings: Mapped["UserSettings"] = relationship(
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    session: Mapped["TelegramSession | None"] = relationship(
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    api_keys: Mapped[list["ApiKey"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class UserSettings(Base):
    __tablename__ = "user_settings"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    auto_reply_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    llm_provider: Mapped[str] = mapped_column(String(16), default="openai")
    use_heavy_model: Mapped[bool] = mapped_column(Boolean, default=False)
    timezone: Mapped[str] = mapped_column(
        String(64), default="UTC"
    )  # IANA tz, например Europe/Moscow
    digest_time: Mapped[str] = mapped_column(
        String(5), default="09:00"
    )  # HH:MM в timezone юзера
    digest_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    transcription_mode: Mapped[str] = mapped_column(
        String(16), default="api"
    )  # local | api | hybrid
    transcription_api_provider: Mapped[str] = mapped_column(
        String(16), default="openai"
    )  # openai | gemini | mistral
    auto_reply_cooldown_min: Mapped[int] = mapped_column(Integer, default=30)
    auto_reply_mode: Mapped[str] = mapped_column(
        String(8), default="static"
    )  # static | smart
    auto_reply_text: Mapped[str] = mapped_column(
        Text,
        default="Сейчас не у телефона, отвечу как только смогу.",
    )
    ignore_archived: Mapped[bool] = mapped_column(Boolean, default=True)
    reminders_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    reminder_lead_hours: Mapped[int] = mapped_column(Integer, default=2)
    reminder_overdue_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    news_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    news_window_hours: Mapped[int] = mapped_column(Integer, default=24)
    news_digest_time: Mapped[str] = mapped_column(
        String(5), default="08:00"
    )  # HH:MM в UTC
    auto_sync_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    auto_sync_interval_sec: Mapped[int] = mapped_column(Integer, default=7200)
    auto_extract_memories: Mapped[bool] = mapped_column(Boolean, default=False)
    include_saved_messages: Mapped[bool] = mapped_column(Boolean, default=False)
    smart_digest_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    smart_digest_interval_min: Mapped[int] = mapped_column(Integer, default=30)
    urgent_notify_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    smart_digest_last_sent: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True
    )
    draft_suggestions_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    draft_only_important: Mapped[bool] = mapped_column(Boolean, default=True)
    draft_max_per_hour: Mapped[int] = mapped_column(Integer, default=5)
    monitored_folders: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # JSON: ["Работа", "Семья"]
    monitor_only_selected_folders: Mapped[bool] = mapped_column(Boolean, default=False)

    # Inbox / auto‑mode
    auto_mode: Mapped[str] = mapped_column(
        String(16), default="offline_only"
    )  # offline_only | always | smart
    quiet_hours_start: Mapped[str | None] = mapped_column(String(5), nullable=True)
    quiet_hours_end: Mapped[str | None] = mapped_column(String(5), nullable=True)
    auto_reply_close_contacts: Mapped[bool] = mapped_column(Boolean, default=True)
    notify_on_auto_reply: Mapped[bool] = mapped_column(Boolean, default=True)

    user: Mapped[User] = relationship(back_populates="settings")


class TelegramSession(Base):
    __tablename__ = "telegram_sessions"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    api_id: Mapped[int] = mapped_column(BigInteger)
    api_hash_enc: Mapped[str] = mapped_column(Text)
    session_string_enc: Mapped[str] = mapped_column(Text)
    phone: Mapped[str] = mapped_column(String(32))
    account_label: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    user: Mapped[User] = relationship(back_populates="session")


class ApiKey(Base):
    __tablename__ = "api_keys"
    __table_args__ = (
        UniqueConstraint("user_id", "provider", name="uq_api_key_user_provider"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    provider: Mapped[str] = mapped_column(String(16))
    key_enc: Mapped[str] = mapped_column(Text)

    user: Mapped[User] = relationship(back_populates="api_keys")


class Contact(Base):
    """Сохранённый профиль чата/контакта (peer)."""

    __tablename__ = "contacts"
    __table_args__ = (
        UniqueConstraint("user_id", "peer_id", name="uq_contact_user_peer"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    peer_id: Mapped[int] = mapped_column(BigInteger, index=True)
    peer_kind: Mapped[str] = mapped_column(String(16))  # user | chat | channel
    is_bot: Mapped[bool] = mapped_column(Boolean, default=False)
    is_archived: Mapped[bool] = mapped_column(
        Boolean, default=False
    )  # лежит в архиве в Telegram
    is_news_source: Mapped[bool] = mapped_column(
        Boolean, default=False
    )  # помеченные для /news каналы
    display_name: Mapped[str] = mapped_column(String(256))
    username: Mapped[str | None] = mapped_column(String(128), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(32), nullable=True)
    style_profile: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # JSON-строка
    style_updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_seen_message_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    folder_names: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # comma-separated folder titles
    archetype: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # null | "close_friend" | "family" | "colleague" | "acquaintance" | "toxic" | "romantic" | "unknown"


class Message(Base):
    """Кэш сообщений из чатов."""

    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint(
            "user_id", "peer_id", "message_id", name="uq_msg_user_peer_id"
        ),
        Index("ix_messages_user_peer_date", "user_id", "peer_id", "date"),
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

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    peer_id: Mapped[int] = mapped_column(BigInteger, index=True)
    peer_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    message_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    direction: Mapped[str] = mapped_column(String(8))  # mine | theirs
    text: Mapped[str] = mapped_column(Text)
    deadline_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(
        String(16), default="open"
    )  # open | done | cancelled | reminded
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class AutoReplyLog(Base):
    """Лог авто-ответов для прозрачности."""

    __tablename__ = "auto_reply_logs"

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


class Memory(Base):
    """Факты о владельце и контактах, извлекаемые из переписок и разговоров с ботом."""

    __tablename__ = "memories"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    contact_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, index=True
    )
    fact: Mapped[str] = mapped_column(Text)
    sentiment: Mapped[str | None] = mapped_column(
        String(16), nullable=True
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
        Boolean, default=True
    )  # активен / опровергнут
    cluster_topic: Mapped[str | None] = mapped_column(
        String(128), nullable=True
    )  # тема-кластер
    embedding_hash: Mapped[str | None] = mapped_column(
        String(16), nullable=True, index=True
    )  # хеш для дедупликации
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    validity_start: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    validity_end: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    importance: Mapped[float] = mapped_column(Float, default=0.5)  # 0.0–1.0
    decay_rate: Mapped[float] = mapped_column(Float, default=0.07)  # скорость забывания
    memory_tier: Mapped[int] = mapped_column(
        Integer, default=1
    )  # 1=эпизод, 2=недельное, 3=месячное
    related_memory_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, index=True
    )  # ссылка на другой Memory.id
    relation_type: Mapped[str | None] = mapped_column(
        String(16), nullable=True
    )  # cause, effect, contradicts, supports, continues, example_of


class MemoryLink(Base):
    """Many-to-many связи между фактами памяти с весами."""

    __tablename__ = "memory_links"

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
        String(16), nullable=True
    )  # cause/effect/contradicts/supports/continues
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


class AgentCache(Base):
    """Кэш результатов сабагентов."""

    __tablename__ = "agent_cache"

    cache_key: Mapped[str] = mapped_column(String(128), primary_key=True)
    result_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    ttl_seconds: Mapped[int] = mapped_column(Integer, default=0)


class ConversationState(Base):
    """Состояние диалога с контактом: непрочитанные, последние события."""

    __tablename__ = "conversation_states"
    __table_args__ = (
        UniqueConstraint("user_id", "peer_id", name="uq_convstate_user_peer"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    peer_id: Mapped[int] = mapped_column(BigInteger, index=True)
    status: Mapped[str] = mapped_column(
        String(16), default="active"
    )  # active | waiting_reply | archived
    unread_count: Mapped[int] = mapped_column(Integer, default=0)
    last_incoming_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_outgoing_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_auto_reply_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


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
