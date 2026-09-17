"""add user avatar_path

Revision ID: 5586124f9a41
Revises: 4a6d4feb9865
Create Date: 2026-09-13 16:44:00.000000

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "5586124f9a41"
down_revision = "4a6d4feb9865"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("avatar_path", sa.String(length=255), nullable=True))


def downgrade() -> None:
    op.drop_column("users", "avatar_path")
