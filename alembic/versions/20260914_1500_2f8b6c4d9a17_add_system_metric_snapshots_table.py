"""add system_metric_snapshots table

Revision ID: 2f8b6c4d9a17
Revises: 9c2e5b7a1f34
Create Date: 2026-09-14 15:00:00.000000

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "2f8b6c4d9a17"
down_revision = "9c2e5b7a1f34"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "system_metric_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("cpu_percent", sa.Float(), nullable=False),
        sa.Column("mem_used_bytes", sa.BigInteger(), nullable=False),
        sa.Column("mem_total_bytes", sa.BigInteger(), nullable=False),
        sa.Column("disk_used_bytes", sa.BigInteger(), nullable=False),
        sa.Column("disk_total_bytes", sa.BigInteger(), nullable=False),
        sa.Column("load_avg_1m", sa.Float(), nullable=True),
        sa.Column("polled_at", sa.DateTime(), nullable=False),
    )
    # Every dashboard query is a bounded polled_at range scan (see app/services/system_metrics_service.py) —
    # same reasoning as telemetry_events' own index.
    op.create_index("ix_system_metric_snapshots_polled_at", "system_metric_snapshots", ["polled_at"])


def downgrade() -> None:
    op.drop_index("ix_system_metric_snapshots_polled_at", table_name="system_metric_snapshots")
    op.drop_table("system_metric_snapshots")
