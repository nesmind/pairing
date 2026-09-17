"""add message status and updated_at

Revision ID: 1fce38adc63a
Revises: e3c5cd6ab5c5
Create Date: 2026-09-11 08:32:00.000000

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "1fce38adc63a"
down_revision = "e3c5cd6ab5c5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # batch mode — see 20260909_1418_a366a6ef0a7b_add_message_sender.py's
    # upgrade() for why: SQLite has no ALTER TABLE support for adding a
    # NOT NULL column to an existing table outside batch mode.
    # server_default='complete' backfills every existing row (every
    # message that predates this feature already finished generating,
    # by definition), so the column can be NOT NULL from day one with no
    # separate data-migration step.
    with op.batch_alter_table("messages", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("status", sa.String(length=20), nullable=False, server_default="complete"),
        )
        batch_op.add_column(sa.Column("updated_at", sa.DateTime(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("messages", schema=None) as batch_op:
        batch_op.drop_column("updated_at")
        batch_op.drop_column("status")
