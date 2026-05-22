"""add anti_ai settings to user_settings

Revision ID: f1a2b3c4d5e6
Revises: e5f6a7b8c9d0
Create Date: 2026-05-23
"""

revision: str = "f1a2b3c4d5e6"
down_revision: str | tuple[str, ...] | None = "e5f6a7b8c9d0"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None

from alembic import op
import sqlalchemy as sa


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    cols = {c["name"] for c in inspector.get_columns("user_settings")}
    if "anti_ai_enabled" not in cols:
        op.add_column(
            "user_settings",
            sa.Column(
                "anti_ai_enabled",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("1"),
            ),
        )
    if "anti_ai_mode" not in cols:
        op.add_column(
            "user_settings",
            sa.Column(
                "anti_ai_mode", sa.String(16), nullable=False, server_default="off"
            ),
        )


def downgrade() -> None:
    op.drop_column("user_settings", "anti_ai_mode")
    op.drop_column("user_settings", "anti_ai_enabled")
