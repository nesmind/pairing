"""
ORM (database table) definitions, split one class-group per file (see
CLAUDE.md's file-size rule) instead of one large models.py. Every class
is re-exported here so the rest of the app keeps writing
`from app.models import User, Conversation, ...` exactly as before —
this package boundary is an internal reorganization, not a change to
any call site outside it.

Import order matters: Alembic (see alembic/env.py) and app.database.init_db
just need every model registered on Base.metadata before comparing
against the live database, which happens as a side effect of importing
this package — the actual order below doesn't affect correctness.
"""

from app.models._base import ID_LEN, new_id, utcnow
from app.models.app_setting import SYSTEM_OWNER_ID, AppSetting
from app.models.channel import Channel, ChannelMember
from app.models.conversation import Conversation, Message, MessageAttachment
from app.models.document import Chunk, Document
from app.models.image_generation import ImageGenerationJob
from app.models.note import Note, NotePin
from app.models.system_metrics import GpuMetricSnapshot, SystemMetricSnapshot
from app.models.telemetry import OllamaModelSnapshot, TelemetryEvent
from app.models.user import User

__all__ = [
    "ID_LEN",
    "new_id",
    "utcnow",
    "SYSTEM_OWNER_ID",
    "AppSetting",
    "Channel",
    "ChannelMember",
    "Conversation",
    "Message",
    "MessageAttachment",
    "Chunk",
    "Document",
    "ImageGenerationJob",
    "Note",
    "NotePin",
    "GpuMetricSnapshot",
    "OllamaModelSnapshot",
    "SystemMetricSnapshot",
    "TelemetryEvent",
    "User",
]
