"""add shared_from column to messages

Revision ID: b3f1a7d9c2e4
Revises: 5c9d3e7a2b48
Create Date: 10:00:00.000000

Marks a message shared into a channel from a private chat (app.services.message_share_service). Existing rows stay null.
"""

import sqlalchemy as sa

from alembic import op

revision = "b3f1a7d9c2e4"
down_revision = "5c9d3e7a2b48"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("shared_from", sa.String(20), nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "shared_from")
