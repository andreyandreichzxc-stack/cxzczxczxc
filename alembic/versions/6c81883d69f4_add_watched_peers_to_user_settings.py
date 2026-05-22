"""add_watched_peers_to_user_settings

Revision ID: 6c81883d69f4
Revises: 318404aba419
Create Date: 2026-05-22 15:24:36.596471

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "6c81883d69f4"
down_revision: Union[str, Sequence[str], None] = "318404aba419"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    cols = {c["name"] for c in inspector.get_columns("user_settings")}
    if "watched_peers" not in cols:
        op.add_column(
            "user_settings", sa.Column("watched_peers", sa.Text(), nullable=True)
        )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("user_settings", "watched_peers")
