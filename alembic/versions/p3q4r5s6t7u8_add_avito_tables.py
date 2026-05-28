"""Add Avito monitoring tables

Revision ID: p3q4r5s6t7u8
Revises: o1p2q3r4s5t6
Create Date: 2026-05-25
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "p3q4r5s6t7u8"
down_revision: Union[str, Sequence[str], None] = "o1p2q3r4s5t6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing = inspector.get_table_names()

    if "avito_listings" not in existing:
        op.create_table(
            "avito_listings",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("user_id", sa.BigInteger(), nullable=False),
            sa.Column("avito_id", sa.String(128), nullable=False),
            sa.Column("search_query", sa.String(512), nullable=False),
            sa.Column("title", sa.String(512), nullable=False),
            sa.Column("price", sa.Integer(), nullable=True),
            sa.Column("url", sa.Text(), nullable=False),
            sa.Column("image_url", sa.Text(), nullable=True),
            sa.Column("city", sa.String(128), nullable=True),
            sa.Column("condition", sa.String(64), nullable=True),
            sa.Column("delivery", sa.Boolean(), server_default=sa.text("0")),
            sa.Column("seller_name", sa.String(256), nullable=True),
            sa.Column("seller_rating", sa.Float(), nullable=True),
            sa.Column("seller_reviews", sa.Integer(), nullable=True),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("deal_score", sa.Integer(), nullable=True),
            sa.Column("is_suspicious", sa.Boolean(), server_default=sa.text("0")),
            sa.Column("scam_reasons", sa.Text(), nullable=True),
            sa.Column("is_active", sa.Boolean(), server_default=sa.text("1")),
            sa.Column(
                "first_seen_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
            ),
            sa.Column(
                "last_seen_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
            ),
            sa.Column("price_changed_at", sa.DateTime(timezone=True), nullable=True),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("user_id", "avito_id", name="uq_avito_user_listing"),
        )
        op.create_index(
            "ix_avito_user_search", "avito_listings", ["user_id", "search_query"]
        )
        op.create_index("ix_avito_last_seen", "avito_listings", ["last_seen_at"])
        op.create_index("ix_avito_listings_avito_id", "avito_listings", ["avito_id"])

    if "avito_price_history" not in existing:
        op.create_table(
            "avito_price_history",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("listing_id", sa.Integer(), nullable=False),
            sa.Column("price", sa.Integer(), nullable=False),
            sa.Column(
                "recorded_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
            ),
            sa.ForeignKeyConstraint(
                ["listing_id"], ["avito_listings.id"], ondelete="CASCADE"
            ),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "ix_avito_price_listing",
            "avito_price_history",
            ["listing_id", "recorded_at"],
        )

    if "avito_watches" not in existing:
        op.create_table(
            "avito_watches",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("user_id", sa.BigInteger(), nullable=False),
            sa.Column("listing_id", sa.Integer(), nullable=False),
            sa.Column("price_threshold", sa.Integer(), nullable=True),
            sa.Column("is_active", sa.Boolean(), server_default=sa.text("1")),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.func.now(),
            ),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(
                ["listing_id"], ["avito_listings.id"], ondelete="CASCADE"
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "user_id", "listing_id", name="uq_avito_watch_user_listing"
            ),
        )


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing = inspector.get_table_names()

    if "avito_watches" in existing:
        op.drop_table("avito_watches")
    if "avito_price_history" in existing:
        op.drop_table("avito_price_history")
    if "avito_listings" in existing:
        op.drop_table("avito_listings")
