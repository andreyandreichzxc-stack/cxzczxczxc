"""Add conversation_summaries table for persistent conversation context.

Revision ID: n2o3p4q5r6s7
Revises: i2j3k4l5m6n7
Create Date: 2026-05-24 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "n2o3p4q5r6s7"
down_revision: Union[str, Sequence[str], None] = "i2j3k4l5m6n7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create conversation_summaries table."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    if "conversation_summaries" not in inspector.get_table_names():
        op.create_table(
            "conversation_summaries",
            sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
            sa.Column("user_id", sa.BigInteger(), nullable=False),
            sa.Column("last_peer_id", sa.BigInteger(), nullable=True),
            sa.Column("last_peer_name", sa.String(256), nullable=True),
            sa.Column("summary_text", sa.Text(), nullable=False),
            sa.Column("turn_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("(datetime('now'))"),
            ),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "ix_conversation_summaries_user_id",
            "conversation_summaries",
            ["user_id"],
        )


def downgrade() -> None:
    """Drop conversation_summaries table."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    if "conversation_summaries" in inspector.get_table_names():
        op.drop_table("conversation_summaries")
