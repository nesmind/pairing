"""add message status and updated_at

Revision ID: 1fce38adc63a
Revises: e3c5cd6ab5c5
Create Date: 08:32:00.000000

"""

import sqlalchemy as sa

from alembic import op

revision = "1fce38adc63a"
down_revision = "e3c5cd6ab5c5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # batch mode: SQLite cannot add a NOT NULL column otherwise. server_default='complete'
    # backfills old messages (all finished) with no data migration.
    with op.batch_alter_table("messages", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("status", sa.String(length=20), nullable=False, server_default="complete"),
        )
        batch_op.add_column(sa.Column("updated_at", sa.DateTime(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("messages", schema=None) as batch_op:
        batch_op.drop_column("updated_at")
        batch_op.drop_column("status")
