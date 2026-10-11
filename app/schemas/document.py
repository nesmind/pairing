"""Request/response shapes for app/routers/documents.py and the RAG limits in app/routers/settings.py."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ScopeSummary(BaseModel):
    """Document/chunk/size counts for one knowledge-base scope (KnowledgeSummary)."""

    document_count: int
    chunk_count: int
    used_mb: float = 0.0


class RagLimits(BaseModel):
    """Admin RAG settings (Settings > System): whether the knowledge base is on, the biggest single upload, and
    the storage cap per user's private documents. `enabled` is checked first by GET /api/settings/rag-availability,
    independent of engine or embedding model. The size limits apply in routers/documents.py uploads, not to the
    admin-only global scope.
    """

    enabled: bool = True
    max_file_mb: float = Field(..., gt=0)
    max_user_space_mb: float = Field(..., gt=0)


class KnowledgeSummary(BaseModel):
    """What Settings shows about the knowledge base: the user's private documents, shared documents and the limits."""

    folder: str
    mine: ScopeSummary
    shared: ScopeSummary
    limits: RagLimits


class DocumentOut(BaseModel):
    """One row of the Knowledge tab list: a display name (not the storage path) and its scope."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    scope: str
    chunk_count: int
    created_at: datetime


class UploadResult(BaseModel):
    """Final outcome of an upload job (UploadJobStatus.result)."""

    summary: KnowledgeSummary
    uploaded: list[str]
    errors: list[str]


class UploadJobStatus(BaseModel):
    """Progress of one upload job. POST /api/documents/upload returns one at once (`done: false` unless all
    files were rejected); the browser polls GET /api/documents/upload/{job_id} until `done`, when `result` is set.
    A plain per-filename dict, as an internal structure, not a public contract.

    `files` maps name to:
      {"status": "waiting"|"processing"|"added"|"updated"|"skipped"|"error",
       "chunk": int, "chunks_total": int|None, "error": str (only if status=="error")}
    """

    job_id: str
    done: bool
    files: dict[str, dict]
    result: UploadResult | None = None
