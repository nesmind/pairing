"""add mode and strength to image_generation_jobs

Revision ID: c4f8a1d6e2b9
Revises: b3e7d9a2c5f8
"""

import sqlalchemy as sa

from alembic import op

revision = "c4f8a1d6e2b9"
down_revision = "b3e7d9a2c5f8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "image_generation_jobs",
        sa.Column("mode", sa.String(20), nullable=False, server_default="text_to_image"),
    )
    op.add_column("image_generation_jobs", sa.Column("strength", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("image_generation_jobs", "strength")
    op.drop_column("image_generation_jobs", "mode")
