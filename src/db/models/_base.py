"""Core ORM: DeclarativeBase and root User entity."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, DateTime, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

if TYPE_CHECKING:
    from src.db.models._auth import ApiKey, LlmKeySlot, TelegramSession, UserSettings
    from src.db.models._messaging import ConversationSummary


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
    conversation_summaries: Mapped[list["ConversationSummary"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
