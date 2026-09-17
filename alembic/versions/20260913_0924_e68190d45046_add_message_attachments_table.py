"""add message_attachments table

Revision ID: e68190d45046
Revises: 1bb757574207
Create Date: 2026-09-13 09:24:00.000000

"""

import uuid

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "e68190d45046"
down_revision = "1bb757574207"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # A message can now carry several attachments (see
    # app.config.MAX_ATTACHMENT_DOCUMENTS/MAX_ATTACHMENT_IMAGES) — this
    # table replaces messages.attachment_path/attachment_filename/
    # attachment_type (one row per file instead of three scalar columns).
    op.create_table(
        "message_attachments",
        sa.Column("id", sa.String(length=32), primary_key=True),
        sa.Column("message_id", sa.String(length=32), sa.ForeignKey("messages.id"), nullable=False),
        sa.Column("path", sa.String(length=500), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("type", sa.String(length=20), nullable=False),
    )

    # Carries forward any message that already has a single attachment
    # under the old scalar columns — a real, if small, live-data concern:
    # this app already has real attachments in production. Done as raw
    # SQL rather than the ORM since Alembic migrations run outside the
    # app's own model layer.
    conn = op.get_bind()
    messages = sa.table(
        "messages",
        sa.column("id", sa.String),
        sa.column("attachment_path", sa.String),
        sa.column("attachment_filename", sa.String),
        sa.column("attachment_type", sa.String),
    )
    message_attachments = sa.table(
        "message_attachments",
        sa.column("id", sa.String),
        sa.column("message_id", sa.String),
        sa.column("path", sa.String),
        sa.column("filename", sa.String),
        sa.column("type", sa.String),
    )
    rows = conn.execute(
        sa.select(
            messages.c.id, messages.c.attachment_path, messages.c.attachment_filename, messages.c.attachment_type
        ).where(messages.c.attachment_path.isnot(None))
    ).fetchall()
    for row in rows:
        conn.execute(
            message_attachments.insert().values(
                id=uuid.uuid4().hex,
                message_id=row.id,
                path=row.attachment_path,
                filename=row.attachment_filename,
                type=row.attachment_type,
            )
        )

    with op.batch_alter_table("messages", schema=None) as batch_op:
        batch_op.drop_column("attachment_path")
        batch_op.drop_column("attachment_filename")
        batch_op.drop_column("attachment_type")


def downgrade() -> None:
    with op.batch_alter_table("messages", schema=None) as batch_op:
        batch_op.add_column(sa.Column("attachment_path", sa.String(length=500), nullable=True))
        batch_op.add_column(sa.Column("attachment_filename", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("attachment_type", sa.String(length=20), nullable=True))

    conn = op.get_bind()
    messages = sa.table(
        "messages",
        sa.column("id", sa.String),
        sa.column("attachment_path", sa.String),
        sa.column("attachment_filename", sa.String),
        sa.column("attachment_type", sa.String),
    )
    message_attachments = sa.table(
        "message_attachments",
        sa.column("id", sa.String),
        sa.column("message_id", sa.String),
        sa.column("path", sa.String),
        sa.column("filename", sa.String),
        sa.column("type", sa.String),
    )
    # Only the first attachment per message survives a downgrade — the
    # old schema only ever had room for one.
    rows = conn.execute(
        sa.select(
            message_attachments.c.message_id,
            message_attachments.c.path,
            message_attachments.c.filename,
            message_attachments.c.type,
        )
    ).fetchall()
    seen_message_ids = set()
    for row in rows:
        if row.message_id in seen_message_ids:
            continue
        seen_message_ids.add(row.message_id)
        conn.execute(
            messages.update()
            .where(messages.c.id == row.message_id)
            .values(attachment_path=row.path, attachment_filename=row.filename, attachment_type=row.type)
        )

    op.drop_table("message_attachments")
