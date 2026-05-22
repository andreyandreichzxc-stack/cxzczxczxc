"""add_pattern_caching_setting

Revision ID: d1c2e3f4a5b6
Revises: fe658c1e6a41
Create Date: 2026-05-22 17:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d1c2e3f4a5b6"
down_revision: Union[str, Sequence[str], None] = "fe658c1e6a41"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema — add pattern_caching_enabled to user_settings."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    cols = {c["name"] for c in inspector.get_columns("user_settings")}
    if "pattern_caching_enabled" not in cols:
        op.add_column(
            "user_settings",
            sa.Column(
                "pattern_caching_enabled",
                sa.Boolean(),
                nullable=False,
                server_default="0",
            ),
        )


def downgrade() -> None:
    """Downgrade schema — remove pattern_caching_enabled from user_settings."""
    op.drop_column("user_settings", "pattern_caching_enabled")
