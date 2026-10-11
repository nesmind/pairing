"""add message reply_to_message_id

Revision ID: 2744265a0922
Revises: 1fce38adc63a
Create Date: 09:00:00.000000

"""

import sqlalchemy as sa

from alembic import op

revision = "2744265a0922"
down_revision = "1fce38adc63a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Self-referential FK (assistant message -> triggering user message): batch mode + explicit
    # name, as in 20260909_1418_a366a6ef0a7b_add_message_sender.py.
    with op.batch_alter_table("messages", schema=None) as batch_op:
        batch_op.add_column(sa.Column("reply_to_message_id", sa.String(length=32), nullable=True))
        batch_op.create_foreign_key(
            "fk_messages_reply_to_message_id_messages", "messages", ["reply_to_message_id"], ["id"]
        )


def downgrade() -> None:
    with op.batch_alter_table("messages", schema=None) as batch_op:
        batch_op.drop_constraint("fk_messages_reply_to_message_id_messages", type_="foreignkey")
        batch_op.drop_column("reply_to_message_id")
