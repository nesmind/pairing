"""Upload validation, staging and the poll-based background jobs behind POST /api/documents/upload
(chunk+embed: document_ingest.py).

Two steps, not one long request: the router validates and saves files, starts the slow chunk+embed via
run_upload_job and returns a job id; the browser polls its status. It used to stream over SSE, which
depends on fetch+ReadableStream behaving the same in every browser; plain POST/GET JSON does.
"""

import re
import time
import uuid
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import INSTANCE_INDEX, KNOWLEDGE_DIR
from app.database import AsyncSessionLocal
from app.models import Document, User
from app.schemas import KnowledgeSummary, ScopeSummary, UploadJobStatus, UploadResult
from app.services.document_extract import SUPPORTED_EXTENSIONS
from app.services.document_ingest import GLOBAL_DIR, user_folder
from app.services.document_sync import sync_folder_stream
from app.services.settings_service import get_rag_limits

# Client filenames are untrusted, but only path separators (already stripped by Path(name).name) and
# characters invalid on some filesystem need blocking. Spaces, parentheses, commas and accented or non-Latin
# letters (e.g. Hebrew) are legitimate and kept, so names look like what the user picked.
_UNSAFE_CHARS_RE = re.compile(r'[\x00-\x1f\x7f<>:"/\\|?*]')


def safe_filename(name: str) -> str:
    base = Path(name or "").name
    base = _UNSAFE_CHARS_RE.sub("_", base).strip().strip(".")
    return base or "file"


async def _scope_summary(db: AsyncSession, owner_id: str | None) -> ScopeSummary:
    docs = (await db.execute(select(Document).where(Document.owner_id == owner_id))).scalars().all()
    return ScopeSummary(
        document_count=len(docs),
        chunk_count=sum(d.chunk_count for d in docs),
        used_mb=round(sum(d.size_bytes for d in docs) / (1024 * 1024), 2),
    )


async def get_knowledge_summary(db: AsyncSession, user: User) -> KnowledgeSummary:
    return KnowledgeSummary(
        folder=str(KNOWLEDGE_DIR),
        mine=await _scope_summary(db, user.id),
        shared=await _scope_summary(db, None),
        limits=await get_rag_limits(db),
    )


# Upload jobs
# In-memory, per-process, not shared across local instances (instance_service.py). A poll on another
# instance would 404, so start_upload_job prefixes job ids with INSTANCE_INDEX and instance_proxy.py's
# balancer routes polls back to the right instance. Jobs are small; _prune_old_jobs bounds the dict.

_upload_jobs: dict[str, dict] = {}
_JOB_TTL_SECONDS = 30 * 60


def _prune_old_jobs() -> None:
    cutoff = time.time() - _JOB_TTL_SECONDS
    stale = [jid for jid, job in _upload_jobs.items() if job["done"] and job["created_at"] < cutoff]
    for jid in stale:
        del _upload_jobs[jid]


def get_job(job_id: str) -> dict | None:
    return _upload_jobs.get(job_id)


def get_job_for_user(job_id: str, user: User) -> dict | None:
    """Like get_job, but None if `user` didn't start it and isn't an admin: a job id is a UUID but visible to the
    browser (network tab, logs), so it must not let others read upload filenames and error text.
    routers/documents.py 404s either way.
    """
    job = get_job(job_id)
    if job is None or (job["user_id"] != user.id and user.role != "admin"):
        return None
    return job


def job_status(job_id: str, job: dict) -> UploadJobStatus:
    return UploadJobStatus(job_id=job_id, done=job["done"], files=job["files"], result=job["result"])


async def _run_upload_job(
    job_id: str,
    target_dir: Path,
    owner_id: str | None,
    accepted: list[str],
    rejects: list[tuple[str, str]],
    user_id: str,
) -> None:
    """The slow chunk + embed work, as a fire-and-forget task after the request returned. Uses its own DB
    session (the request's is closed by then).
    """
    job = _upload_jobs[job_id]
    accepted_set = set(accepted)
    extra_errors: list[str] = []
    stats_errors: list[str] = []
    async with AsyncSessionLocal() as db:
        try:
            async for event in sync_folder_stream(db, target_dir, owner_id):
                filename = event.get("filename")
                if event["stage"] == "done":
                    stats_errors = event["stats"]["errors"]
                elif filename not in accepted_set:
                    # sync_folder_stream re-checks the owner's *whole* folder; other up-to-date documents aren't part of
                    # this job's report.
                    continue
                elif event["stage"] == "file_start":
                    job["files"][filename] = {"status": "processing", "chunk": 0, "chunks_total": None}
                elif event["stage"] == "chunk_progress":
                    job["files"][filename].update(chunk=event["chunk"], chunks_total=event["chunks_total"])
                elif event["stage"] == "file_done":
                    entry = job["files"].setdefault(filename, {})
                    entry["status"] = event["status"]
                    if event["status"] == "error":
                        entry["error"] = event["error"]
        except Exception as exc:  # noqa: BLE001 - a bug here must still resolve the job, not hang it forever
            extra_errors.append(f"Upload job hit an unexpected error partway through: {exc}")

        user = await db.get(User, user_id)
        result = UploadResult(
            summary=await get_knowledge_summary(db, user),
            uploaded=accepted,
            errors=[f"{filename}: {reason}" for filename, reason in rejects] + stats_errors + extra_errors,
        )
        job["result"] = result.model_dump()
    job["done"] = True


async def stage_upload(
    db: AsyncSession,
    files: list[UploadFile],
    scope: str,
    user: User,
) -> tuple[list[str], list[tuple[str, str]], Path, str | None]:
    """Validates each file against admin limits (size, type, per-user quota) and writes accepted ones to disk
    (the fast synchronous half). Returns (accepted filenames, (name, rejection reason) pairs, target folder,
    owner_id) for run_upload_job / start_upload_job.

    A failing, unsupported or unparseable file (e.g. a corrupt or password-protected PDF) is its own per-file
    error; the rest of the batch continues.
    """
    limits = await get_rag_limits(db)
    max_file_bytes = limits.max_file_mb * 1024 * 1024
    quota_bytes = limits.max_user_space_mb * 1024 * 1024

    target_dir = GLOBAL_DIR if scope == "global" else user_folder(user.id)
    owner_id = None if scope == "global" else user.id

    # Running usage is tracked per accepted file, so a batch can partly succeed. Re-uploading counts only the
    # new size against the quota (the old one is subtracted first).
    existing_sizes_by_name: dict[str, int] = {}
    running_total = 0
    if scope == "mine":
        existing_docs = (await db.execute(select(Document).where(Document.owner_id == user.id))).scalars().all()
        existing_sizes_by_name = {Path(d.filename).name: d.size_bytes for d in existing_docs}
        running_total = sum(existing_sizes_by_name.values())

    accepted: list[str] = []
    rejects: list[tuple[str, str]] = []
    for upload in files:
        name = safe_filename(upload.filename)
        if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            rejects.append((upload.filename, "unsupported file type."))
            continue

        raw = await upload.read()
        if len(raw) > max_file_bytes:
            rejects.append((upload.filename, f"exceeds the {limits.max_file_mb:g}MB file size limit."))
            continue

        if scope == "mine":
            projected = running_total - existing_sizes_by_name.get(name, 0) + len(raw)
            if projected > quota_bytes:
                rejects.append((upload.filename, f"would exceed your {limits.max_user_space_mb:g}MB storage limit."))
                continue
            running_total = projected

        (target_dir / name).write_bytes(raw)
        accepted.append(name)

    return accepted, rejects, target_dir, owner_id


async def start_upload_job(
    db: AsyncSession,
    background_tasks,
    target_dir: Path,
    owner_id: str | None,
    accepted: list[str],
    rejects: list[tuple[str, str]],
    user: User,
) -> UploadJobStatus:
    """Registers a job and, if anything is left to process, schedules run_upload_job. Returns the initial status
    snapshot to poll (see module docstring).
    """
    _prune_old_jobs()
    # Prefixed with this process's INSTANCE_INDEX so instance_proxy.py's balancer routes later polls to the
    # instance holding the job's in-memory _upload_jobs entry (never shared across instances).
    job_id = f"{INSTANCE_INDEX}-{uuid.uuid4().hex}"
    files_state = {name: {"status": "waiting", "chunk": 0, "chunks_total": None} for name in accepted}
    for filename, reason in rejects:
        files_state[filename] = {"status": "error", "error": reason}
    _upload_jobs[job_id] = {
        "created_at": time.time(),
        "done": not accepted,  # nothing to do if every file was rejected
        "files": files_state,
        "result": None,
        # Who started the job; routers/documents.py GET /upload/{job_id} checks it so users can't poll others' jobs.
        "user_id": user.id,
    }
    if accepted:
        background_tasks.add_task(_run_upload_job, job_id, target_dir, owner_id, accepted, rejects, user.id)
    else:
        _upload_jobs[job_id]["result"] = UploadResult(
            summary=await get_knowledge_summary(db, user),
            uploaded=[],
            errors=[f"{filename}: {reason}" for filename, reason in rejects],
        ).model_dump()

    return job_status(job_id, _upload_jobs[job_id])
