"""add channels

Revision ID: 7bbd7485d7d8
Revises: 6d4212846f5c
Create Date: 2026-09-09 12:28:08.622647

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "7bbd7485d7d8"
down_revision = "6d4212846f5c"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "channels",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_table(
        "channel_members",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("channel_id", sa.String(length=32), nullable=False),
        sa.Column("user_id", sa.String(length=32), nullable=False),
        sa.Column("is_manager", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["channel_id"],
            ["channels.id"],
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("channel_id", "user_id", name="uq_channel_member_channel_user"),
    )
    # batch mode (rather than plain op.add_column/create_unique_constraint/
    # create_foreign_key): SQLite has no ALTER TABLE support for adding a
    # constraint to an existing table — only batch mode's copy-and-move
    # strategy can do it there. Explicit constraint names (rather than
    # autogenerate's None) so downgrade() can drop them by name on every
    # backend, including MySQL where an unnamed drop_constraint doesn't
    # resolve on its own.
    with op.batch_alter_table("conversations", schema=None) as batch_op:
        batch_op.add_column(sa.Column("channel_id", sa.String(length=32), nullable=True))
        batch_op.create_unique_constraint("uq_conversations_channel_id", ["channel_id"])
        batch_op.create_foreign_key(
            "fk_conversations_channel_id_channels",
            "channels",
            ["channel_id"],
            ["id"],
        )


def downgrade() -> None:
    with op.batch_alter_table("conversations", schema=None) as batch_op:
        batch_op.drop_constraint("fk_conversations_channel_id_channels", type_="foreignkey")
        batch_op.drop_constraint("uq_conversations_channel_id", type_="unique")
        batch_op.drop_column("channel_id")
    op.drop_table("channel_members")
    op.drop_table("channels")
