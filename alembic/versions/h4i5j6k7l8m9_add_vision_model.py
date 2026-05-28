"""add vision_model to user_settings

Revision ID: h4i5j6k7l8m9
Revises: g3h4i5j6k7l8
Create Date: 2026-05-28
"""

revision: str = "h4i5j6k7l8m9"
down_revision: str | tuple[str, ...] | None = "g3h4i5j6k7l8"
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None

from alembic import op
import sqlalchemy as sa


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    cols = {c["name"] for c in inspector.get_columns("user_settings")}
    if "vision_model" not in cols:
        op.add_column(
            "user_settings",
            sa.Column(
                "vision_model",
                sa.String(100),
                nullable=True,
            ),
        )


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    cols = {c["name"] for c in inspector.get_columns("user_settings")}
    if "vision_model" in cols:
        op.drop_column("user_settings", "vision_model")
