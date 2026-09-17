"""
Knowledge-base endpoints: every user has their own private documents
(uploaded here, up to MAX_UPLOAD_FILES at a time), plus a shared
"global" scope that only an admin can add to but every user's chats can
draw on. See app/services/document_upload.py for the upload
validation/job orchestration and app/services/document_ingest.py for the
chunk+embed pipeline itself — this file only handles HTTP routing,
status codes, and Depends() injection.

Files dropped directly onto the server filesystem instead of uploading
through the browser are still picked up automatically the next time the
app starts (see app.services.document_ingest.sync_all_on_startup) —
just not live, since there's no "Rescan" button anymore.
"""

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import KNOWLEDGE_DIR
from app.database import get_db
from app.models import Document, User
from app.schemas import DocumentOut, KnowledgeSummary, OkResponse, UploadJobStatus
from app.services import document_upload
from app.services.auth_service import get_current_user
from app.services.document_ingest import MAX_UPLOAD_FILES

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.get("/summary", response_model=KnowledgeSummary)
async def get_summary(db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    return await document_upload.get_knowledge_summary(db, user)


@router.get("", response_model=list[DocumentOut])
async def list_documents(db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    """Every document the current user's chats can actually retrieve
    from: their own private uploads, plus every shared one."""
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
    """Validates and saves up to MAX_UPLOAD_FILES documents into either
    the current user's own private folder ("mine", the default —
    everyone can do this) or the shared "global" folder ("global" —
    admin-only, since it affects every user's chats), then hands the
    slow chunk+embed work off to a background task and returns
    immediately with a job id. Poll GET /upload/{job_id} for progress."""
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
    """Polled by the browser every second or two while a job is running
    (see app/static/js/settings.js) — deliberately just a snapshot read,
    no side effects, so polling it is always safe to retry. See
    document_upload.get_job_for_user's own docstring for the ownership
    check behind this 404."""
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
    """Deletes one document and its chunks — your own private one, or a
    shared one if you're admin. 404s (not 403) for anything else, the
    same convention used for conversations, so a regular user can't use
    this to even confirm whether some other id exists."""
    document = await db.get(Document, document_id)
    owns_it = document is not None and document.owner_id == user.id
    can_delete_shared = document is not None and document.owner_id is None and user.role == "admin"
    if document is None or not (owns_it or can_delete_shared):
        raise HTTPException(status_code=404, detail="Document not found.")

    # Remove the file first: if something goes wrong between the two
    # steps, worst case is a file gone with its row still present, which
    # the next sync self-heals (sees it as "removed" and deletes the
    # row) — the other order could leave the file behind to be silently
    # re-ingested as if it were new.
    (KNOWLEDGE_DIR / document.filename).unlink(missing_ok=True)
    await db.delete(document)
    await db.commit()
    return OkResponse()
