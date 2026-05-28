"""Add SkillOpt-inspired fields to skills table

Revision ID: q4r5s6t7u8v9
Revises: p3q4r5s6t7u8
Create Date: 2026-05-26
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "q4r5s6t7u8v9"
down_revision: Union[str, Sequence[str], None] = "p3q4r5s6t7u8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add V2 skill evolution fields: version, edit_history, rejected_edits, validation_score, best_body."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = {col["name"] for col in inspector.get_columns("skills")}

    # version field (semver string)
    if "version" not in columns:
        op.add_column(
            "skills",
            sa.Column("version", sa.String(32), nullable=False, server_default="1.0.0"),
        )

    # edit_history_json (JSON array of edit records)
    if "edit_history_json" not in columns:
        op.add_column(
            "skills", sa.Column("edit_history_json", sa.JSON(), nullable=True)
        )

    # rejected_edits_json (JSON array of rejected edits for negative feedback)
    if "rejected_edits_json" not in columns:
        op.add_column(
            "skills", sa.Column("rejected_edits_json", sa.JSON(), nullable=True)
        )

    # validation_score (float, last validation gate score)
    if "validation_score" not in columns:
        op.add_column(
            "skills", sa.Column("validation_score", sa.Float(), nullable=True)
        )

    # best_body (text, best-performing body snapshot)
    if "best_body" not in columns:
        op.add_column("skills", sa.Column("best_body", sa.Text(), nullable=True))


def downgrade() -> None:
    """Remove V2 skill evolution fields."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = {col["name"] for col in inspector.get_columns("skills")}

    for col_name in [
        "best_body",
        "validation_score",
        "rejected_edits_json",
        "edit_history_json",
        "version",
    ]:
        if col_name in columns:
            op.drop_column("skills", col_name)
