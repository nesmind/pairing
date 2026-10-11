"""Document and Chunk models for the RAG knowledge base (logic: app/services/document_service.py)."""

from sqlalchemy import Column, DateTime, ForeignKey, Integer, LargeBinary, String, Text
from sqlalchemy.orm import relationship

from app.database import Base
from app.models._base import ID_LEN, new_id, utcnow


class Document(Base):
    """A knowledge-base file (KNOWLEDGE_DIR), chunked and embedded. A derived index kept in sync with disk
    by document_service; also reachable via upload (app/routers/documents.py).
    """

    __tablename__ = "documents"

    id = Column(String(ID_LEN), primary_key=True, default=new_id)
    # Owner: NULL = shared "global" folder (admin upload, visible to everyone's retrieval, see
    # document_service.retrieve_context); a user id = that user's private upload.
    owner_id = Column(String(ID_LEN), ForeignKey("users.id"), nullable=True)
    # Path relative to KNOWLEDGE_DIR ("global/handbook.pdf", "users/<id>/notes.txt"); unique, and the
    # file's identity for sync (document_service._sync_folder).
    filename = Column(String(500), nullable=False, unique=True)
    # SHA-256 of the file at last ingestion; only a changed hash triggers re-embedding.
    content_hash = Column(String(64), nullable=False)
    # Number of chunks produced (shown in the UI).
    chunk_count = Column(Integer, default=0)
    # File size in bytes, for the per-user RAG quota (app/services/document_upload.py) without re-reading disk.
    size_bytes = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=utcnow)

    # lazy="selectin" (see Conversation.messages); cascade delete and document.chunks rely on it.
    chunks = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class Chunk(Base):
    """One embedded slice of a Document's text, for similarity search."""

    __tablename__ = "chunks"

    id = Column(String(ID_LEN), primary_key=True, default=new_id)
    document_id = Column(String(ID_LEN), ForeignKey("documents.id"), nullable=False)
    text = Column(Text, nullable=False)
    # Embedding vector as raw bytes (serialized float32 numpy array). A BLOB with a brute-force cosine scan
    # (document_service.py) suffices at on-prem scale; see ROADMAP.md for a vector store.
    embedding = Column(LargeBinary, nullable=False)

    # lazy="selectin": retrieve_context reads chunk.document.filename (see Conversation.messages).
    document = relationship("Document", back_populates="chunks", lazy="selectin")
