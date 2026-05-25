"""Add memory_digest to ContactProfile.

Revision ID: h1i2j3k4l5m6
Revises: g0h1i2j3k4l5
Create Date: 2026-05-24
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "h1i2j3k4l5m6"
down_revision: Union[str, Sequence[str], None] = "g0h1i2j3k4l5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add memory_digest and memory_digest_updated_at columns to contact_profiles."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    cols = {c["name"] for c in inspector.get_columns("contact_profiles")}

    if "memory_digest" not in cols:
        op.add_column(
            "contact_profiles",
            sa.Column("memory_digest", sa.Text(), nullable=True),
        )
    if "memory_digest_updated_at" not in cols:
        op.add_column(
            "contact_profiles",
            sa.Column(
                "memory_digest_updated_at",
                sa.DateTime(timezone=True),
                nullable=True,
            ),
        )


def downgrade() -> None:
    """Remove memory_digest columns from contact_profiles."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    cols = {c["name"] for c in inspector.get_columns("contact_profiles")}

    if "memory_digest_updated_at" in cols:
        op.drop_column("contact_profiles", "memory_digest_updated_at")
    if "memory_digest" in cols:
        op.drop_column("contact_profiles", "memory_digest")
