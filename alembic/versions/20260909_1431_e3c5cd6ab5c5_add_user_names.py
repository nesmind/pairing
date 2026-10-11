"""add user names

Revision ID: e3c5cd6ab5c5
Revises: a366a6ef0a7b
Create Date: 14:31:57.750880

"""

import sqlalchemy as sa

from alembic import op

revision = "e3c5cd6ab5c5"
down_revision = "a366a6ef0a7b"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("first_name", sa.String(length=100), nullable=True))
    op.add_column("users", sa.Column("last_name", sa.String(length=100), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "last_name")
    op.drop_column("users", "first_name")
