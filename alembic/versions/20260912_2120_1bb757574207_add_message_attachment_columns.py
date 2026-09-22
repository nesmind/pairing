"""add message attachment columns

Revision ID: 1bb757574207
Revises: 2744265a0922
Create Date: 2026-09-12 21:20:16.584369

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "1bb757574207"
down_revision = "2744265a0922"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Only ever set on a "user"-role message sent with a file attached
    # (see app.services.chat_attachment_service) — plain nullable columns,
    # no FK/batch mode needed (unlike reply_to_message_id's self-
    # referential FK in the prior migration).
    with op.batch_alter_table("messages", schema=None) as batch_op:
        batch_op.add_column(sa.Column("attachment_path", sa.String(length=500), nullable=True))
        batch_op.add_column(sa.Column("attachment_filename", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("attachment_type", sa.String(length=20), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("messages", schema=None) as batch_op:
        batch_op.drop_column("attachment_type")
        batch_op.drop_column("attachment_filename")
        batch_op.drop_column("attachment_path")
