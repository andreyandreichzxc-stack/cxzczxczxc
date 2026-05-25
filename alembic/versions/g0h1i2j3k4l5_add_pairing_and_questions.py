"""Add allowed_contacts and pending_questions tables.

Revision ID: g0h1i2j3k4l5
Revises: f1a2b3c4d5e6
Create Date: 2026-05-24
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "g0h1i2j3k4l5"
down_revision: Union[str, None] = "f1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())

    if "allowed_contacts" not in tables:
        op.create_table(
            "allowed_contacts",
            sa.Column("telegram_id", sa.BigInteger(), nullable=False),
            sa.Column(
                "approved_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("CURRENT_TIMESTAMP"),
            ),
            sa.Column("label", sa.String(256), nullable=True),
            sa.PrimaryKeyConstraint("telegram_id"),
        )
    if "pending_questions" not in tables:
        op.create_table(
            "pending_questions",
            sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
            sa.Column("owner_id", sa.BigInteger(), nullable=False),
            sa.Column("question", sa.String(512), nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("CURRENT_TIMESTAMP"),
            ),
            sa.PrimaryKeyConstraint("id"),
        )
    indexes = {idx["name"] for idx in inspector.get_indexes("pending_questions")}
    if "ix_pending_questions_owner_id" not in indexes:
        op.create_index(
            "ix_pending_questions_owner_id", "pending_questions", ["owner_id"]
        )


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())
    if "pending_questions" in tables:
        op.drop_table("pending_questions")
    if "allowed_contacts" in tables:
        op.drop_table("allowed_contacts")
