"""add_last_compressed_at_to_skills

Revision ID: c6c5965acc9d
Revises: r1s2t3u4v5w6
Create Date: 2026-05-26 22:37:04.847391

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c6c5965acc9d"
down_revision: Union[str, Sequence[str], None] = "r1s2t3u4v5w6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add last_compressed_at column to skills table."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = {col["name"] for col in inspector.get_columns("skills")}

    if "last_compressed_at" not in columns:
        op.add_column(
            "skills",
            sa.Column("last_compressed_at", sa.DateTime(), nullable=True),
        )


def downgrade() -> None:
    """Remove last_compressed_at column from skills table."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = {col["name"] for col in inspector.get_columns("skills")}

    if "last_compressed_at" in columns:
        op.drop_column("skills", "last_compressed_at")
