"""add telemetry_events and ollama_model_snapshots tables

Revision ID: 9c2e5b7a1f34
Revises: 5586124f9a41
Create Date: 2026-09-14 12:00:00.000000

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "9c2e5b7a1f34"
down_revision = "5586124f9a41"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "telemetry_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("kind", sa.String(length=20), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=True),
        sa.Column("host", sa.String(length=255), nullable=False),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("duration_ms", sa.Float(), nullable=False),
        sa.Column("total_duration_ms", sa.Float(), nullable=True),
        sa.Column("load_duration_ms", sa.Float(), nullable=True),
        sa.Column("prompt_eval_duration_ms", sa.Float(), nullable=True),
        sa.Column("eval_duration_ms", sa.Float(), nullable=True),
        sa.Column("prompt_eval_count", sa.Integer(), nullable=True),
        sa.Column("eval_count", sa.Integer(), nullable=True),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
    )
    # Every dashboard query for this table is a bounded started_at range
    # scan (see app/services/telemetry_service.py) — only cheap with an
    # index, unlike the plain-PK-only image_generation_jobs table this
    # migration's style is otherwise copied from.
    op.create_index("ix_telemetry_events_started_at", "telemetry_events", ["started_at"])

    op.create_table(
        "ollama_model_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("host", sa.String(length=255), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("size_vram_bytes", sa.BigInteger(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("polled_at", sa.DateTime(), nullable=False),
    )
    op.create_index(
        "ix_ollama_model_snapshots_host_model_polled",
        "ollama_model_snapshots",
        ["host", "model_name", "polled_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_ollama_model_snapshots_host_model_polled", table_name="ollama_model_snapshots")
    op.drop_table("ollama_model_snapshots")
    op.drop_index("ix_telemetry_events_started_at", table_name="telemetry_events")
    op.drop_table("telemetry_events")
