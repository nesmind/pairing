"""add messages.completed_at (when an assistant reply ended; shown as its time)

Revision ID: f7c3e5a9b2d4
Revises: e6b2d4f8a1c3
"""

import sqlalchemy as sa

from alembic import op

revision = "f7c3e5a9b2d4"
down_revision = "e6b2d4f8a1c3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Existing replies keep null: chat.js falls back to created_at for them.
    with op.batch_alter_table("messages") as batch:
        batch.add_column(sa.Column("completed_at", sa.DateTime(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("messages") as batch:
        batch.drop_column("completed_at")
