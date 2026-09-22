"""add image_generation_jobs table

Revision ID: 4a6d4feb9865
Revises: e68190d45046
Create Date: 2026-09-13 11:56:00.000000

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "4a6d4feb9865"
down_revision = "e68190d45046"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "image_generation_jobs",
        sa.Column("id", sa.String(length=32), primary_key=True),
        sa.Column("owner_id", sa.String(length=32), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("negative_prompt", sa.Text(), nullable=True),
        sa.Column("checkpoint", sa.String(length=255), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("steps", sa.Integer(), nullable=False),
        sa.Column("cfg", sa.Float(), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("comfy_prompt_id", sa.String(length=64), nullable=True),
        sa.Column("image_path", sa.String(length=500), nullable=True),
        sa.Column("image_filename", sa.String(length=255), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("image_generation_jobs")
