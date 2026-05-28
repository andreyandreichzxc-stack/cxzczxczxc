"""Merge memory refactor and LLM refactor heads

Revision ID: e1d7c0f3ac9c
Revises: h4i5j6k7l8m9, c6c5965acc9d
Create Date: 2026-05-28 18:34:21.688903

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e1d7c0f3ac9c'
down_revision: Union[str, Sequence[str], None] = ('h4i5j6k7l8m9', 'c6c5965acc9d')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
