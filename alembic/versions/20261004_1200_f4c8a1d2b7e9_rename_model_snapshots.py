"""rename ollama_model_snapshots to model_snapshots (it holds Matricxon rows too)

Revision ID: f4c8a1d2b7e9
Revises: e2a9c4d7b1f6
"""

from alembic import op

revision = "f4c8a1d2b7e9"
down_revision = "e2a9c4d7b1f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Indexes are dropped and recreated (portable: SQLite has no ALTER INDEX).
    op.drop_index("ix_ollama_model_snapshots_host_model_polled", table_name="ollama_model_snapshots")
    op.drop_index("ix_ollama_model_snapshots_engine", table_name="ollama_model_snapshots")
    op.rename_table("ollama_model_snapshots", "model_snapshots")
    op.create_index("ix_model_snapshots_host_model_polled", "model_snapshots", ["host", "model_name", "polled_at"])
    op.create_index("ix_model_snapshots_engine", "model_snapshots", ["engine"])


def downgrade() -> None:
    op.drop_index("ix_model_snapshots_engine", table_name="model_snapshots")
    op.drop_index("ix_model_snapshots_host_model_polled", table_name="model_snapshots")
    op.rename_table("model_snapshots", "ollama_model_snapshots")
    op.create_index(
        "ix_ollama_model_snapshots_host_model_polled",
        "ollama_model_snapshots",
        ["host", "model_name", "polled_at"],
    )
    op.create_index("ix_ollama_model_snapshots_engine", "ollama_model_snapshots", ["engine"])
