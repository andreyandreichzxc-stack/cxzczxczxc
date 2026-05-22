"""add_style_profile_columns

Revision ID: b8e4f2a3c5d1
Revises: a7c3d9e1f0b2
Create Date: 2026-05-22 16:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b8e4f2a3c5d1"
down_revision: Union[str, Sequence[str], None] = "a7c3d9e1f0b2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema — add style_profile columns to adaptive_personas."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    cols = {c["name"] for c in inspector.get_columns("adaptive_personas")}
    if "style_profile" not in cols:
        op.add_column(
            "adaptive_personas",
            sa.Column("style_profile", sa.Text(), nullable=True),
        )
    if "style_profile_updated_at" not in cols:
        op.add_column(
            "adaptive_personas",
            sa.Column(
                "style_profile_updated_at",
                sa.DateTime(timezone=True),
                nullable=True,
            ),
        )


def downgrade() -> None:
    """Downgrade schema — remove style_profile columns from adaptive_personas."""
    op.drop_column("adaptive_personas", "style_profile_updated_at")
    op.drop_column("adaptive_personas", "style_profile")
