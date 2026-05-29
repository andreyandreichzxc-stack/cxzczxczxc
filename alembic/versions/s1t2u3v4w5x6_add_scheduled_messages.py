"""Add scheduled_messages table for delayed message delivery.

Revision ID: s1t2u3v4w5x6
Revises: r1s2t3u4v5w6
Create Date: 2026-05-29 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "s1t2u3v4w5x6"
down_revision: Union[str, Sequence[str], None] = "r1s2t3u4v5w6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create scheduled_messages table."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    if "scheduled_messages" not in inspector.get_table_names():
        op.create_table(
            "scheduled_messages",
            sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
            sa.Column("user_id", sa.BigInteger(), nullable=False),
            sa.Column("contact_name", sa.String(255), nullable=False),
            sa.Column("text", sa.Text(), nullable=False),
            sa.Column("send_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column(
                "status",
                sa.String(16),
                nullable=False,
                server_default="pending",
            ),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("(datetime('now'))"),
            ),
            sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("error", sa.Text(), nullable=True),
            sa.Column(
                "via_userbot",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("1"),
            ),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "ix_scheduled_messages_user_id",
            "scheduled_messages",
            ["user_id"],
        )


def downgrade() -> None:
    """Drop scheduled_messages table."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    if "scheduled_messages" in inspector.get_table_names():
        op.drop_table("scheduled_messages")
