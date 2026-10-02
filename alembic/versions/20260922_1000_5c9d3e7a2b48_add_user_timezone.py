"""add timezone column to users

Revision ID: 5c9d3e7a2b48
Revises: 8d4e6a2f7c19
Create Date: 2026-09-22 10:00:00.000000

Existing accounts backfill to "UTC" via the server default.
"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "5c9d3e7a2b48"
down_revision = "8d4e6a2f7c19"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("users") as batch:
        batch.add_column(sa.Column("timezone", sa.String(64), nullable=False, server_default="UTC"))


def downgrade() -> None:
    with op.batch_alter_table("users") as batch:
        batch.drop_column("timezone")
