"""add shared_from column to messages

Revision ID: b3f1a7d9c2e4
Revises: 5c9d3e7a2b48
Create Date: 2026-10-03 10:00:00.000000

Marks a message that was shared into a channel from a private chat (see
app.services.message_share_service). No backfill: every existing message was written directly, so null.
"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "b3f1a7d9c2e4"
down_revision = "5c9d3e7a2b48"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("shared_from", sa.String(20), nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "shared_from")
