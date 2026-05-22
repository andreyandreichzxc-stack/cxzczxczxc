"""add_personality_fields

Revision ID: fe658c1e6a41
Revises: 6c81883d69f4
Create Date: 2026-05-22 15:26:33.996329

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "fe658c1e6a41"
down_revision: Union[str, Sequence[str], None] = "6c81883d69f4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema — adds ChatGPT-style personality fields to adaptive_personas."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    cols = {c["name"] for c in inspector.get_columns("adaptive_personas")}
    if "base_tone" not in cols:
        op.add_column(
            "adaptive_personas",
            sa.Column(
                "base_tone",
                sa.String(length=32),
                nullable=False,
                server_default="default",
            ),
        )
    if "warmth" not in cols:
        op.add_column(
            "adaptive_personas",
            sa.Column(
                "warmth", sa.String(length=16), nullable=False, server_default="normal"
            ),
        )
    if "enthusiasm" not in cols:
        op.add_column(
            "adaptive_personas",
            sa.Column(
                "enthusiasm",
                sa.String(length=16),
                nullable=False,
                server_default="normal",
            ),
        )
    if "headings_lists" not in cols:
        op.add_column(
            "adaptive_personas",
            sa.Column(
                "headings_lists",
                sa.String(length=16),
                nullable=False,
                server_default="normal",
            ),
        )
    if "emoji_level" not in cols:
        op.add_column(
            "adaptive_personas",
            sa.Column(
                "emoji_level",
                sa.String(length=16),
                nullable=False,
                server_default="normal",
            ),
        )
    if "custom_instructions" not in cols:
        op.add_column(
            "adaptive_personas",
            sa.Column("custom_instructions", sa.Text(), nullable=True),
        )
    if "alias" not in cols:
        op.add_column(
            "adaptive_personas", sa.Column("alias", sa.String(length=64), nullable=True)
        )
    if "adaptive_mode_enabled" not in cols:
        op.add_column(
            "adaptive_personas",
            sa.Column(
                "adaptive_mode_enabled",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("0"),
            ),
        )
    if "base_snapshot_json" not in cols:
        op.add_column(
            "adaptive_personas",
            sa.Column("base_snapshot_json", sa.Text(), nullable=True),
        )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("adaptive_personas", "base_snapshot_json")
    op.drop_column("adaptive_personas", "adaptive_mode_enabled")
    op.drop_column("adaptive_personas", "alias")
    op.drop_column("adaptive_personas", "custom_instructions")
    op.drop_column("adaptive_personas", "emoji_level")
    op.drop_column("adaptive_personas", "headings_lists")
    op.drop_column("adaptive_personas", "enthusiasm")
    op.drop_column("adaptive_personas", "warmth")
    op.drop_column("adaptive_personas", "base_tone")
