"""An MCP (Model Context Protocol) server an admin connected, offering tools to the model."""

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text

from app.database import Base
from app.models._base import ID_LEN, new_id, utcnow


class McpServer(Base):
    __tablename__ = "mcp_servers"

    id = Column(String(ID_LEN), primary_key=True, default=new_id)
    # Short slug: also the prefix of this server's tool names as the model sees them.
    name = Column(String(40), nullable=False, unique=True)
    url = Column(String(500), nullable=False)  # streamable-HTTP endpoint
    # JSON of request headers (e.g. Authorization), Fernet-encrypted (see secret_crypto).
    headers_encrypted = Column(Text, nullable=True)
    enabled = Column(Boolean, nullable=False, default=True)
    # How long one tool call on this server may take before the model is told it timed out.
    call_timeout_seconds = Column(Integer, nullable=False, default=20, server_default="20")
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
