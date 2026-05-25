"""Contact/peer models: Contact, ContactProfile, ConversationState."""

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
from sqlalchemy.orm import Mapped, mapped_column

from ._base import Base


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

    custom_instructions: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # JSON {"rules": ["rule1", "rule2"]} — per-contact style rules

    memory_digest: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # JSON — precomputed per-contact summary (facts, promises, health, etc.)
    memory_digest_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


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


class AllowedContact(Base):
    """Pairing allowlist — approved contacts that bypass the pairing guard."""

    __tablename__ = "allowed_contacts"

    telegram_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    approved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    label: Mapped[str | None] = mapped_column(String(256), nullable=True)
