"""Add endpoint column to llm_key_slots"""

revision: str = "o1p2q3r4s5t6"
down_revision: str | None = "n2o3p4q5r6s7"

from alembic import op
import sqlalchemy as sa


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = [c["name"] for c in inspector.get_columns("llm_key_slots")]
    if "endpoint" not in columns:
        op.add_column(
            "llm_key_slots", sa.Column("endpoint", sa.String(256), nullable=True)
        )


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = [c["name"] for c in inspector.get_columns("llm_key_slots")]
    if "endpoint" in columns:
        op.drop_column("llm_key_slots", "endpoint")
