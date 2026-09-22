"""add error_message column to messages

Revision ID: 8d4e6a2f7c19
Revises: 4b7f2e91a6c3
Create Date: 2026-09-20 15:00:00.000000

No backfill: every existing status="error" row's real failure reason was only ever published to whoever was
watching the live SSE stream at that moment (see reply_broadcast_service.publish_error) and never persisted, so
there is nothing to recover for rows written before this column existed — they stay null and fall back to
chat.js's generic REPLY_FAILED_NOTICE, exactly as before. Only a new failure written after this migration gets
the real detail.
"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "8d4e6a2f7c19"
down_revision = "4b7f2e91a6c3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("error_message", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "error_message")
