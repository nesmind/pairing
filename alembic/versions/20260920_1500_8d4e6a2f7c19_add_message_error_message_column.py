"""add error_message column to messages

Revision ID: 8d4e6a2f7c19
Revises: 4b7f2e91a6c3
Create Date: 15:00:00.000000

No backfill: old error reasons were only sent over live SSE, never stored; those rows stay null
and show chat.js's generic REPLY_FAILED_NOTICE.
"""

import sqlalchemy as sa

from alembic import op

revision = "8d4e6a2f7c19"
down_revision = "4b7f2e91a6c3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("error_message", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "error_message")
