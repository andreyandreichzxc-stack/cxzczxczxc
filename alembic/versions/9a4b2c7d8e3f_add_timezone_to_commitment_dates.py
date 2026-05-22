"""add_timezone_to_commitment_dates

Add timezone=True to Commitment.deadline_at and Commitment.created_at
for future-aware datetime consistency.

Revision ID: 9a4b2c7d8e3f
Revises: fe658c1e6a41
Create Date: 2026-05-22 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "9a4b2c7d8e3f"
down_revision: Union[str, Sequence[str], None] = "fe658c1e6a41"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    SQLite does not support ALTER COLUMN natively, and create_all() in the
    initial migration already creates these columns with DateTime(timezone=True).
    On SQLite there is no behavioural difference between DateTime and
    DateTime(timezone=True) — both store ISO-8601 strings.
    """
    pass


def downgrade() -> None:
    """Downgrade schema. No-op on SQLite — see upgrade() comment."""
    pass
