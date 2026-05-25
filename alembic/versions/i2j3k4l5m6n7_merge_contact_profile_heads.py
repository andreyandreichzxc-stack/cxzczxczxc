"""Merge contact profile migration heads.

Revision ID: i2j3k4l5m6n7
Revises: c7d8e9f0a1b2, h1i2j3k4l5m6
Create Date: 2026-05-24
"""

from typing import Sequence, Union


# revision identifiers, used by Alembic.
revision: str = "i2j3k4l5m6n7"
down_revision: Union[str, Sequence[str], None] = (
    "c7d8e9f0a1b2",
    "h1i2j3k4l5m6",
)
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Merge-only migration; both parent revisions contain the schema changes."""


def downgrade() -> None:
    """Merge-only migration; downgrade is handled by parent revisions."""
