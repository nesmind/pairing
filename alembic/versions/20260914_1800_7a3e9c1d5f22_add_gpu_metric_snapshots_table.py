"""add gpu_metric_snapshots table

Revision ID: 7a3e9c1d5f22
Revises: 2f8b6c4d9a17
Create Date: 2026-09-14 18:00:00.000000

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "7a3e9c1d5f22"
down_revision = "2f8b6c4d9a17"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "gpu_metric_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("gpu_index", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("utilization_percent", sa.Float(), nullable=False),
        sa.Column("mem_used_bytes", sa.BigInteger(), nullable=False),
        sa.Column("mem_total_bytes", sa.BigInteger(), nullable=False),
        sa.Column("temperature_c", sa.Float(), nullable=True),
        sa.Column("polled_at", sa.DateTime(), nullable=False),
    )
    # Same reasoning as system_metric_snapshots' own index: every dashboard query is a bounded polled_at range
    # scan (see app/services/system_metrics_service.py).
    op.create_index("ix_gpu_metric_snapshots_polled_at", "gpu_metric_snapshots", ["polled_at"])


def downgrade() -> None:
    op.drop_index("ix_gpu_metric_snapshots_polled_at", table_name="gpu_metric_snapshots")
    op.drop_table("gpu_metric_snapshots")
