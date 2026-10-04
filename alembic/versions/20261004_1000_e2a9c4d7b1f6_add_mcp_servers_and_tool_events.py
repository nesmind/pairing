"""add mcp_servers table and messages.tool_events

Revision ID: e2a9c4d7b1f6
Revises: b3f1a7d9c2e4
Create Date: 2026-10-04 10:00:00.000000

MCP tool support: the servers an admin connected, and the tool calls a reply made (shown under it).
"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "e2a9c4d7b1f6"
down_revision = "b3f1a7d9c2e4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "mcp_servers",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("name", sa.String(40), nullable=False, unique=True),
        sa.Column("url", sa.String(500), nullable=False),
        sa.Column("headers_encrypted", sa.Text, nullable=True),
        sa.Column("enabled", sa.Boolean, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.add_column("messages", sa.Column("tool_events", sa.JSON, nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "tool_events")
    op.drop_table("mcp_servers")
