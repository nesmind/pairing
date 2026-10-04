"""add call_timeout_seconds to mcp_servers

Revision ID: a7d2c5e9b3f1
Revises: f4c8a1d2b7e9
"""

import sqlalchemy as sa

from alembic import op

revision = "a7d2c5e9b3f1"
down_revision = "f4c8a1d2b7e9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("mcp_servers", sa.Column("call_timeout_seconds", sa.Integer(), nullable=False, server_default="20"))


def downgrade() -> None:
    op.drop_column("mcp_servers", "call_timeout_seconds")
