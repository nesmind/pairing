"""Request/response shapes for app/routers/documents.py and the RAG
upload-limit settings in app/routers/settings.py."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ScopeSummary(BaseModel):
    """Document/chunk/size counts for one knowledge-base scope (see
    KnowledgeSummary)."""

    document_count: int
    chunk_count: int
    used_mb: float = 0.0


class RagLimits(BaseModel):
    """Admin-configured caps on RAG uploads (see Settings > System):
    the biggest single file anyone may upload, and the most storage one
    user's own private documents may add up to. Enforced in
    app/routers/documents.py's upload handler. Not applied to the
    shared/global scope, which only admin can add to anyway."""

    max_file_mb: float = Field(..., gt=0)
    max_user_space_mb: float = Field(..., gt=0)


class KnowledgeSummary(BaseModel):
    """What Settings shows about the knowledge base: the current user's
    own private documents, the shared documents every user's chats can
    also draw on, and the current upload limits."""

    folder: str
    mine: ScopeSummary
    shared: ScopeSummary
    limits: RagLimits


class DocumentOut(BaseModel):
    """One row in the Knowledge tab's document list — a display name
    (not its full owner-scoped storage path) plus which scope it's in."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    scope: str  # "mine" | "shared"
    chunk_count: int
    created_at: datetime


class UploadResult(BaseModel):
    """The final outcome of an upload job, once done — see
    UploadJobStatus.result below."""

    summary: KnowledgeSummary
    uploaded: list[str]
    errors: list[str]


class UploadJobStatus(BaseModel):
    """Progress snapshot for one upload job. POST /api/documents/upload
    returns one of these immediately (with `done: false` unless every
    file was rejected up front); the browser polls
    GET /api/documents/upload/{job_id} for the same shape until `done`
    is true, at which point `result` is populated. Deliberately a plain
    per-filename dict rather than a typed list — this is an internal
    progress structure, not a stable public API contract.

    `files` maps each file's name to a dict shaped like:
      {"status": "waiting"|"processing"|"added"|"updated"|"skipped"|"error",
       "chunk": int, "chunks_total": int|None, "error": str (only if status=="error")}
    """

    job_id: str
    done: bool
    files: dict[str, dict]
    result: UploadResult | None = None
