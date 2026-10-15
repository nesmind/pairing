"""rename image_generation_jobs.comfy_prompt_id to engine_job_id (ComfyUI was removed; it holds the engine's job id)

Revision ID: e6b2d4f8a1c3
Revises: d5a9c3e7f1b2
"""

import sqlalchemy as sa

from alembic import op

revision = "e6b2d4f8a1c3"
down_revision = "d5a9c3e7f1b2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Batch mode: SQLite rebuilds the table, MySQL renames in place; existing values are kept either way.
    with op.batch_alter_table("image_generation_jobs") as batch:
        batch.alter_column("comfy_prompt_id", new_column_name="engine_job_id", existing_type=sa.String(64))


def downgrade() -> None:
    with op.batch_alter_table("image_generation_jobs") as batch:
        batch.alter_column("engine_job_id", new_column_name="comfy_prompt_id", existing_type=sa.String(64))
