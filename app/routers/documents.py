"""Knowledge-base endpoints: private documents per user (up to MAX_UPLOAD_FILES per upload) plus a shared
"global" scope only admins add to. Upload orchestration: app/services/document_upload.py; chunk+embed:
document_ingest.py.

Files placed on the server filesystem are picked up at next start
(document_ingest.sync_all_on_startup), not live.
"""

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import KNOWLEDGE_DIR
from app.database import get_db
from app.models import Document, User
from app.schemas import DocumentOut, EmbeddingStatus, KnowledgeSummary, OkResponse, ReembedResult, UploadJobStatus
from app.services import document_sync, document_upload
from app.services.auth_service import get_current_user, require_admin
from app.services.document_embedding_status import DocumentEmbeddingStatus
from app.services.document_ingest import MAX_UPLOAD_FILES

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.get("/summary", response_model=KnowledgeSummary)
async def get_summary(db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    return await document_upload.get_knowledge_summary(db, user)


@router.get("/embedding-status", response_model=EmbeddingStatus)
async def embedding_status(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    """How many documents match the current embedding model (Settings > System > RAG)."""
    return await DocumentEmbeddingStatus(db).get()


@router.post("/reembed", response_model=ReembedResult)
async def reembed_all_documents(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    """Embeds every document again with the current model, e.g. after switching embedding models."""
    stats = await document_sync.reembed_all(db)
    status = await DocumentEmbeddingStatus(db).get()
    return ReembedResult(updated=stats["updated"], errors=stats["errors"], status=status)


@router.get("", response_model=list[DocumentOut])
async def list_documents(db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    """Documents the user's chats can retrieve from: own plus shared."""
    docs = (
        (
            await db.execute(
                select(Document)
                .where(or_(Document.owner_id == user.id, Document.owner_id.is_(None)))
                .order_by(Document.created_at.desc()),
            )
        )
        .scalars()
        .all()
    )
    return [
        DocumentOut(
            id=d.id,
            filename=Path(d.filename).name,
            scope="mine" if d.owner_id == user.id else "shared",
            chunk_count=d.chunk_count,
            created_at=d.created_at,
        )
        for d in docs
    ]


@router.post("/upload", response_model=UploadJobStatus)
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    scope: str = Form("mine"),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Validates and saves up to MAX_UPLOAD_FILES documents to the user's folder ("mine", default) or the shared
    folder ("global", admin-only), then embeds in the background and returns a job id. Poll GET /upload/{job_id}.
    """
    if scope not in ("mine", "global"):
        raise HTTPException(status_code=400, detail="scope must be 'mine' or 'global'.")
    if scope == "global" and user.role != "admin":
        raise HTTPException(status_code=403, detail="Only an admin can add shared documents.")
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(status_code=400, detail=f"Upload at most {MAX_UPLOAD_FILES} files at a time.")

    accepted, rejects, target_dir, owner_id = await document_upload.stage_upload(db, files, scope, user)
    return await document_upload.start_upload_job(db, background_tasks, target_dir, owner_id, accepted, rejects, user)


@router.get("/upload/{job_id}", response_model=UploadJobStatus)
def get_upload_job(job_id: str, user: User = Depends(get_current_user)):
    """Polled every second or two while a job runs (settings.js); a side-effect-free snapshot.
    Ownership check: document_upload.get_job_for_user.
    """
    job = document_upload.get_job_for_user(job_id, user)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown or expired upload job.")
    return document_upload.job_status(job_id, job)


@router.delete("/{document_id}", response_model=OkResponse)
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Deletes a document and its chunks (own, or shared if admin). Returns 404, not 403, otherwise, so ids can't be
    probed."""
    document = await db.get(Document, document_id)
    owns_it = document is not None and document.owner_id == user.id
    can_delete_shared = document is not None and document.owner_id is None and user.role == "admin"
    if document is None or not (owns_it or can_delete_shared):
        raise HTTPException(status_code=404, detail="Document not found.")

    # File first: a failure between steps leaves a row without its file, which the next sync cleans; the
    # reverse order would silently re-ingest a leftover file.
    (KNOWLEDGE_DIR / document.filename).unlink(missing_ok=True)
    await db.delete(document)
    await db.commit()
    return OkResponse()
