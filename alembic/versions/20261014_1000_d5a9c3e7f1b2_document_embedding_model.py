"""record which embedding model each knowledge-base document was embedded with

Revision ID: d5a9c3e7f1b2
Revises: c4f8a1d6e2b9
"""

import sqlalchemy as sa

from alembic import op

revision = "d5a9c3e7f1b2"
down_revision = "c4f8a1d6e2b9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # NULL = embedded before this was recorded (model unknown).
    op.add_column("documents", sa.Column("embedding_model", sa.String(255), nullable=True))


def downgrade() -> None:
    op.drop_column("documents", "embedding_model")
