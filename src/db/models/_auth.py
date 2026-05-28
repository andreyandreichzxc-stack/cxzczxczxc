"""Auth models: UserSettings, TelegramSession, ApiKey, LlmKeySlot."""

from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ._base import Base, User


class UserSettings(Base):
    __tablename__ = "user_settings"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    auto_reply_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    llm_provider: Mapped[str] = mapped_column(String(24), default="openai")
    use_heavy_model: Mapped[bool] = mapped_column(Boolean, default=False)
    model_overrides: Mapped[str | None] = mapped_column(
        Text, nullable=True, default=None
    )  # JSON: {"maestro": "deepseek-reasoner", "draft": "deepseek-chat"}
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
    proactive_last_sent: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True
    )
    habit_last_run_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    draft_suggestions_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    draft_only_important: Mapped[bool] = mapped_column(Boolean, default=True)
    draft_max_per_hour: Mapped[int] = mapped_column(Integer, default=5)
    monitored_folders: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # JSON: ["Работа", "Семья"]
    monitor_only_selected_folders: Mapped[bool] = mapped_column(Boolean, default=False)
    watched_peers: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # JSON: [123456, 789012] — peer_id чатов за которыми следим

    # Inbox / auto‑mode
    auto_mode: Mapped[str] = mapped_column(
        String(16), default="offline_only"
    )  # offline_only | always | smart
    quiet_hours_start: Mapped[str | None] = mapped_column(String(5), nullable=True)
    quiet_hours_end: Mapped[str | None] = mapped_column(String(5), nullable=True)
    auto_reply_close_contacts: Mapped[bool] = mapped_column(Boolean, default=True)
    notify_on_auto_reply: Mapped[bool] = mapped_column(Boolean, default=True)
    pattern_caching_enabled: Mapped[bool] = mapped_column(Boolean, default=False)

    # Anti-AI humanizer settings
    anti_ai_enabled: Mapped[bool] = mapped_column(
        Boolean, default=False
    )  # Phase 4 hook
    anti_ai_mode: Mapped[str] = mapped_column(
        String(16), default="off"
    )  # "off" | "log" | "fix"

    # Vision / multimodal
    vision_model: Mapped[str | None] = mapped_column(
        String(100), nullable=True, default=None
    )  # Модель для мультимодального анализа изображений

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
    endpoint: Mapped[str | None] = mapped_column(
        String(256), nullable=True
    )  # custom base_url for OpenAI-compatible APIs
    model: Mapped[str | None] = mapped_column(
        String(128), nullable=True
    )  # конкретная модель (gpt-4o, claude-3-5-sonnet)
    category: Mapped[str] = mapped_column(
        String(16), default="llm", server_default="llm"
    )  # llm | stt | tts | vision
    key_enc: Mapped[str] = mapped_column(Text)
    priority: Mapped[int] = mapped_column(
        Integer, default=0
    )  # чем выше, тем приоритетнее
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    cooldown_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_error: Mapped[str | None] = mapped_column(String(256), nullable=True)
    last_error_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    failure_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    user: Mapped[User] = relationship(back_populates="key_slots")


class PendingQuestion(Base):
    """Pending questions queued during async reply generation."""

    __tablename__ = "pending_questions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    owner_id: Mapped[int] = mapped_column(BigInteger, index=True)
    question: Mapped[str] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
