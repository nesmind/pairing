"""add log_offset to image_generation_jobs

Revision ID: b3e7d9a2c5f8
Revises: a7d2c5e9b3f1
"""

import sqlalchemy as sa

from alembic import op

revision = "b3e7d9a2c5f8"
down_revision = "a7d2c5e9b3f1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("image_generation_jobs", sa.Column("log_offset", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("image_generation_jobs", "log_offset")
