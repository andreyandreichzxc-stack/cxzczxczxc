"""initial_schema

Revision ID: 0ea3133e3615
Revises:
Create Date: 2026-05-20 23:17:22.251602

Note: The initial migration is empty because all ORM tables already exist
in the database (created via Base.metadata.create_all in init_db()).

FTS5 virtual tables (messages_fts*, memories_fts*) are excluded from Alembic's
view — they are created and managed by init_db() via raw SQL in session.py.

Legacy ALTER TABLE operations (columns added to users, memories, commitments,
conversation_states) remain in init_db() for backward compatibility with
existing databases. New schema changes should be added as Alembic migrations.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0ea3133e3615"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
