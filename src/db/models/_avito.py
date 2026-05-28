"""Avito models: AvitoListing, AvitoPriceHistory, AvitoWatch."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ._base import Base, User


class AvitoListing(Base):
    """Scraped Avito listing cache with incremental analysis."""

    __tablename__ = "avito_listings"
    __table_args__ = (
        UniqueConstraint("user_id", "avito_id", name="uq_avito_user_listing"),
        Index("ix_avito_user_search", "user_id", "search_query"),
        Index("ix_avito_last_seen", "last_seen_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    avito_id: Mapped[str] = mapped_column(String(128), index=True)
    search_query: Mapped[str] = mapped_column(String(512))
    title: Mapped[str] = mapped_column(String(512))
    price: Mapped[int | None] = mapped_column(Integer, nullable=True)
    url: Mapped[str] = mapped_column(Text)
    image_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    city: Mapped[str | None] = mapped_column(String(128), nullable=True)
    condition: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )  # new, used, excellent, good, satisfactory
    delivery: Mapped[bool] = mapped_column(Boolean, default=False)
    seller_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    seller_rating: Mapped[float | None] = mapped_column(nullable=True)
    seller_reviews: Mapped[int | None] = mapped_column(Integer, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    deal_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_suspicious: Mapped[bool] = mapped_column(Boolean, default=False)
    scam_reasons: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    price_changed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    price_history: Mapped[list[AvitoPriceHistory]] = relationship(
        back_populates="listing", cascade="all, delete-orphan"
    )
    watches: Mapped[list[AvitoWatch]] = relationship(
        back_populates="listing", cascade="all, delete-orphan"
    )


class AvitoPriceHistory(Base):
    """Price change history for tracked listings."""

    __tablename__ = "avito_price_history"
    __table_args__ = (Index("ix_avito_price_listing", "listing_id", "recorded_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    listing_id: Mapped[int] = mapped_column(
        ForeignKey("avito_listings.id", ondelete="CASCADE"), index=True
    )
    price: Mapped[int] = mapped_column(Integer)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # Relationship
    listing: Mapped[AvitoListing] = relationship(back_populates="price_history")


class AvitoWatch(Base):
    """User watch list for price monitoring."""

    __tablename__ = "avito_watches"
    __table_args__ = (
        UniqueConstraint("user_id", "listing_id", name="uq_avito_watch_user_listing"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    listing_id: Mapped[int] = mapped_column(
        ForeignKey("avito_listings.id", ondelete="CASCADE"), index=True
    )
    price_threshold: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )  # alert when price drops below this
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # Relationship
    listing: Mapped[AvitoListing] = relationship(back_populates="watches")
    user: Mapped[User] = relationship()
