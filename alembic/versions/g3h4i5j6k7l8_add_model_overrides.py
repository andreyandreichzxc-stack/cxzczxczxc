"""add model_overrides to user_settings

Revision ID: g3h4i5j6k7l8
Revises: f1a2b3c4d5e6
Create Date: 2026-05-28
"""

revision: str = "g3h4i5j6k7l8"
down_revision: str | tuple[str, ...] | None = "f1a2b3c4d5e6"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None

from alembic import op
import sqlalchemy as sa


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    cols = {c["name"] for c in inspector.get_columns("user_settings")}
    if "model_overrides" not in cols:
        op.add_column(
            "user_settings",
            sa.Column(
                "model_overrides",
                sa.Text(),
                nullable=True,
            ),
        )


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    cols = {c["name"] for c in inspector.get_columns("user_settings")}
    if "model_overrides" in cols:
        op.drop_column("user_settings", "model_overrides")
