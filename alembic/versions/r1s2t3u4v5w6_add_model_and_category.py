"""Add model and category columns to llm_key_slots"""

revision: str = "r1s2t3u4v5w6"
down_revision: str | None = "q4r5s6t7u8v9"

from alembic import op
import sqlalchemy as sa


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = [c["name"] for c in inspector.get_columns("llm_key_slots")]
    if "model" not in columns:
        op.add_column(
            "llm_key_slots", sa.Column("model", sa.String(128), nullable=True)
        )
    if "category" not in columns:
        op.add_column(
            "llm_key_slots",
            sa.Column(
                "category",
                sa.String(16),
                nullable=False,
                server_default="llm",
            ),
        )


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = [c["name"] for c in inspector.get_columns("llm_key_slots")]
    if "category" in columns:
        op.drop_column("llm_key_slots", "category")
    if "model" in columns:
        op.drop_column("llm_key_slots", "model")
