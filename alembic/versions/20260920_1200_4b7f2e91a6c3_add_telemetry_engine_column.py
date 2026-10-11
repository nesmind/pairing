"""add engine column to telemetry_events and ollama_model_snapshots

Revision ID: 4b7f2e91a6c3
Revises: 7a3e9c1d5f22
Create Date: 12:00:00.000000

Existing rows are Ollama's (the only earlier writers); server_default="ollama" backfills them.
"""

import sqlalchemy as sa

from alembic import op

revision = "4b7f2e91a6c3"
down_revision = "7a3e9c1d5f22"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "telemetry_events",
        sa.Column("engine", sa.String(length=20), nullable=False, server_default="ollama"),
    )
    op.create_index("ix_telemetry_events_engine", "telemetry_events", ["engine"])

    op.add_column(
        "ollama_model_snapshots",
        sa.Column("engine", sa.String(length=20), nullable=False, server_default="ollama"),
    )
    op.create_index("ix_ollama_model_snapshots_engine", "ollama_model_snapshots", ["engine"])


def downgrade() -> None:
    op.drop_index("ix_ollama_model_snapshots_engine", table_name="ollama_model_snapshots")
    op.drop_column("ollama_model_snapshots", "engine")
    op.drop_index("ix_telemetry_events_engine", table_name="telemetry_events")
    op.drop_column("telemetry_events", "engine")
