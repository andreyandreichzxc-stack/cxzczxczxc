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
    func,
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
    key_slots: Mapped[list["LlmKeySlot"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="select",
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


class LlmKeySlot(Base):
    """Слот LLM-ключа — один ключ для конкретного провайдера и назначения."""

    __tablename__ = "llm_key_slots"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    provider: Mapped[str] = mapped_column(String(16))  # openai/gemini/mistral
    purpose: Mapped[str] = mapped_column(
        String(32), default="main"
    )  # main/draft/memory/background/search/analysis/urgent/fallback
    label: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )  # человекочитаемая метка "основной", "для черновиков"
    key_enc: Mapped[str] = mapped_column(Text)
    priority: Mapped[int] = mapped_column(
        Integer, default=0
    )  # чем выше, тем приоритетнее
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    cooldown_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_error: Mapped[str | None] = mapped_column(String(256), nullable=True)
    last_error_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    failure_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    user: Mapped[User] = relationship(back_populates="key_slots")


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


class ContactProfile(Base):
    """Профиль контакта — не просто archetype, а полноценная карточка."""

    __tablename__ = "contact_profiles"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    contact_id: Mapped[int] = mapped_column(BigInteger, index=True)

    closeness: Mapped[float] = mapped_column(Float, default=0.5)  # 0-1 близость
    closeness_label: Mapped[str | None] = mapped_column(
        String(32), nullable=True
    )  # друг/работа/семья/клиент

    communication_style: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # "мягко и коротко", "деловито"
    key_topics: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # JSON ["ремонт", "дети", "работа"]
    sensitivity: Mapped[float] = mapped_column(
        Float, default=0.5
    )  # 0-1 чувствительность к тону
    communication_dos: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # JSON ["писать утром", "без голосовых"]
    communication_donts: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # JSON ["не критиковать", "не слать ночью"]

    current_status: Mapped[str | None] = mapped_column(
        String(16), nullable=True
    )  # active/tension/resolved/distant
    last_emotional_event: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_emotional_event_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True
    )

    relationship_phase: Mapped[str | None] = mapped_column(
        String(16), nullable=True
    )  # warming/cooling/stable
    open_questions: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # JSON ["договориться о встрече", "вернуть долг"]

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


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
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
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
        Index(
            "ix_conversation_states_user_active",
            "user_id",
            "status",
            "last_incoming_at",
        ),
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
    radar_snoozed_until: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
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


class SelfProfile(Base):
    """Память о владельце — предпочтения, цели, проекты, стиль."""

    __tablename__ = "self_profile"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), unique=True, index=True
    )

    preferences: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # JSON ["чай", "утренние созвоны", ...]
    goals: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # JSON ["закончить проект X", ...]
    current_projects: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    decision_style: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # "быстрый"/"аналитический"/"советуется"
    communication_preferences: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # JSON
    sleep_pattern: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # "сова"/"жаворонок"/"00:00-08:00"
    work_hours: Mapped[str | None] = mapped_column(
        String(32), nullable=True
    )  # "09:00-18:00"

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
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


class InstructionProfile(Base):
    """Профиль инструкций — активные правила поведения бота."""

    __tablename__ = "instruction_profiles"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    rules_json: Mapped[str] = mapped_column(Text, default="[]")  # JSON список правил
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class InstructionCandidate(Base):
    """Кандидат в инструкции — предложенное правило, ждёт подтверждения."""

    __tablename__ = "instruction_candidates"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    rule: Mapped[str] = mapped_column(Text)  # текст правила
    category: Mapped[str] = mapped_column(
        String(32), default="tone"
    )  # tone/format/privacy/memory/agent/llm_suggestion/consolidation/conflict
    is_safe: Mapped[bool] = mapped_column(
        Boolean, default=False
    )  # безопасное → авто-применить
    llm_reviewed: Mapped[bool] = mapped_column(
        Boolean, default=False
    )  # обработано LLM-оптимизатором
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class InstructionEvent(Base):
    """Событие — когда пользователь дал обратную связь."""

    __tablename__ = "instruction_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    raw_text: Mapped[str] = mapped_column(Text)  # что сказал пользователь
    detected_rule: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # какое правило извлекли
    action: Mapped[str] = mapped_column(
        String(16), default="detected"
    )  # detected/applied/asked/ignored
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


class AdaptivePersona(Base):
    """Адаптивный профиль личности бота — стиль общения подстраивается под пользователя."""

    __tablename__ = "adaptive_personas"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), unique=True, index=True
    )
    # Стиль
    brevity: Mapped[str] = mapped_column(
        String(16), default="normal"
    )  # short/normal/detailed
    formality: Mapped[str] = mapped_column(
        String(16), default="friendly"
    )  # formal/friendly/casual
    emoji_usage: Mapped[str] = mapped_column(
        String(16), default="normal"
    )  # none/minimal/normal/rich
    initiative: Mapped[str] = mapped_column(
        String(16), default="reactive"
    )  # reactive/proactive/balanced
    # Формат
    preferred_format: Mapped[str] = mapped_column(
        String(16), default="text"
    )  # text/bullets/numbered
    use_html: Mapped[bool] = mapped_column(Boolean, default=True)
    max_response_len: Mapped[int] = mapped_column(
        Integer, default=500
    )  # макс символов в ответе
    # Запреты
    forbidden_patterns: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # JSON-список запретов
    # Режимы
    quiet_hours_active: Mapped[bool] = mapped_column(Boolean, default=False)
    work_mode: Mapped[str] = mapped_column(
        String(16), default="normal"
    )  # normal/focus/relax
    # Метрики
    total_interactions: Mapped[int] = mapped_column(Integer, default=0)
    total_corrections: Mapped[int] = mapped_column(Integer, default=0)
    last_correction_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
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


class SoulSnapshot(Base):
    """Снапшот tier-2 soul-блоков для версионирования промптов."""

    __tablename__ = "soul_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(32), nullable=False)  # semver "1.0.0"
    snapshot_type: Mapped[str] = mapped_column(
        String(16), nullable=False, default="auto"
    )  # manual / auto / freeze
    blocks_json: Mapped[dict] = mapped_column(JSON, nullable=False)  # все tier-2 блоки
    diff_from_previous: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    approved_by: Mapped[str] = mapped_column(
        String(64), nullable=False, default="system"
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Trajectory(Base):
    """Recorded assistant turn for learning, debugging, and skill extraction."""

    __tablename__ = "trajectories"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    request_text: Mapped[str] = mapped_column(Text, nullable=False)
    route_mode: Mapped[str | None] = mapped_column(
        String(32), nullable=True, index=True
    )
    intent_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    actions_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    used_skills_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    memory_ids_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    response_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), index=True
    )


class Skill(Base):
    """Prompt-level procedural memory. V1 skills are hints, not executable code."""

    __tablename__ = "skills"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    trigger_patterns_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    review_status: Mapped[str] = mapped_column(
        String(16), default="approved", index=True
    )  # approved | pending | rejected
    success_count: Mapped[int] = mapped_column(Integer, default=0)
    failure_count: Mapped[int] = mapped_column(Integer, default=0)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class SkillUsage(Base):
    """Skill application telemetry linked to a trajectory."""

    __tablename__ = "skill_usages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    skill_id: Mapped[int] = mapped_column(
        ForeignKey("skills.id", ondelete="CASCADE"), index=True
    )
    trajectory_id: Mapped[int | None] = mapped_column(
        ForeignKey("trajectories.id", ondelete="SET NULL"), nullable=True, index=True
    )
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), index=True
    )
