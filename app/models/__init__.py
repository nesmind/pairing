"""ORM definitions, one class-group per file, re-exported here so `from app.models import User,...` works.

Importing the package registers every model on Base.metadata (needed by Alembic and app.database.init_db).
"""

from app.models._base import ID_LEN, new_id, utcnow
from app.models.app_setting import SYSTEM_OWNER_ID, AppSetting
from app.models.channel import Channel, ChannelMember
from app.models.conversation import Conversation, Message, MessageAttachment
from app.models.document import Chunk, Document
from app.models.image_generation import ImageGenerationJob
from app.models.mcp_server import McpServer
from app.models.note import Note, NotePin
from app.models.system_metrics import GpuMetricSnapshot, SystemMetricSnapshot
from app.models.telemetry import ModelSnapshot, TelemetryEvent
from app.models.user import User

__all__ = [
    "McpServer",
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
    "ModelSnapshot",
    "SystemMetricSnapshot",
    "TelemetryEvent",
    "User",
]
