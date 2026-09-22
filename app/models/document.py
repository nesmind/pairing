"""Document + Chunk models for the RAG knowledge base — see
app/services/document_service.py for the ingestion/retrieval logic built
on top of these."""

from sqlalchemy import Column, DateTime, ForeignKey, Integer, LargeBinary, String, Text
from sqlalchemy.orm import relationship

from app.database import Base
from app.models._base import ID_LEN, new_id, utcnow


class Document(Base):
    """One file from the knowledge base (app.config.KNOWLEDGE_DIR) that's
    been chunked and embedded for RAG. Rows here are kept in sync with
    what's actually on disk by app.services.document_service's sync
    functions — this table is a derived index, not a place users manage
    files directly (though they can also get here via upload — see
    app/routers/documents.py)."""

    __tablename__ = "documents"

    id = Column(String(ID_LEN), primary_key=True, default=new_id)
    # Whose document this is: NULL means it's in the shared "global"
    # folder (uploaded by an admin, visible to every user's RAG
    # retrieval — see app.services.document_service.retrieve_context); a
    # real user id means it's that user's own private upload, visible
    # only to them.
    owner_id = Column(String(ID_LEN), ForeignKey("users.id"), nullable=True)
    # Path relative to KNOWLEDGE_DIR, e.g. "global/handbook.pdf" or
    # "users/<user_id>/notes.txt" — unique across the whole tree since
    # each owner has their own subfolder, and doubles as the file's
    # identity for sync purposes (see
    # app.services.document_service._sync_folder).
    filename = Column(String(500), nullable=False, unique=True)
    # SHA-256 of the file's bytes at last ingestion (64 hex chars), used
    # to detect an edited file on the next sync without needing to
    # re-read+re-embed every file on every scan — only a changed hash
    # triggers re-ingestion.
    content_hash = Column(String(64), nullable=False)
    # Total chunks produced from this document (informational, shown in
    # the UI so the user can see ingestion actually happened).
    chunk_count = Column(Integer, default=0)
    # Raw file size in bytes, used to enforce the admin-configured
    # per-user RAG storage quota (see app/services/document_upload.py's
    # upload handler) without re-reading every file from disk on every
    # check.
    size_bytes = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=utcnow)

    # lazy="selectin": see app/models/conversation.py's Conversation.messages
    # for why commonly-accessed relationships default to this under the
    # app's async session — cascade-deleting a Document's chunks and
    # reading document.chunks both rely on it being safely loadable.
    chunks = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class Chunk(Base):
    """One embedded slice of a Document's text, used for similarity search."""

    __tablename__ = "chunks"

    id = Column(String(ID_LEN), primary_key=True, default=new_id)
    document_id = Column(String(ID_LEN), ForeignKey("documents.id"), nullable=False)
    text = Column(Text, nullable=False)
    # The embedding vector, stored as raw bytes (a serialized numpy
    # float32 array). Kept as a BLOB rather than a separate vector
    # database because, at the scale of a single on-prem deployment,
    # a brute-force cosine-similarity scan in
    # app/services/document_service.py is fast enough and needs zero
    # extra infrastructure — see ROADMAP.md for when to swap this out
    # for a real vector store.
    embedding = Column(LargeBinary, nullable=False)

    # lazy="selectin" — retrieve_context reads chunk.document.filename
    # for every scored chunk; see Conversation.messages for why this is
    # the app's default loading strategy for accessed relationships.
    document = relationship("Document", back_populates="chunks", lazy="selectin")
