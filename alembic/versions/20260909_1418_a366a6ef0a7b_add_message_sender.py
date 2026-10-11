"""add message sender

Revision ID: a366a6ef0a7b
Revises: 7bbd7485d7d8
Create Date: 14:18:48.668597

"""

import sqlalchemy as sa

from alembic import op

revision = "a366a6ef0a7b"
down_revision = "7bbd7485d7d8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # batch mode + explicit constraint name: see 20260909_1228_7bbd7485d7d8_add_channels.py.
    with op.batch_alter_table("messages", schema=None) as batch_op:
        batch_op.add_column(sa.Column("sender_id", sa.String(length=32), nullable=True))
        batch_op.create_foreign_key("fk_messages_sender_id_users", "users", ["sender_id"], ["id"])


def downgrade() -> None:
    with op.batch_alter_table("messages", schema=None) as batch_op:
        batch_op.drop_constraint("fk_messages_sender_id_users", type_="foreignkey")
        batch_op.drop_column("sender_id")
