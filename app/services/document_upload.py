"""
Upload validation, on-disk staging, and the poll-based background-job orchestration behind POST
/api/documents/upload — the business logic that used to live directly in app/routers/documents.py. See
app/services/document_ingest.py for the actual chunk+embed pipeline this kicks off once files are accepted.

Uploading is a two-step, poll-based flow rather than one long request: the router validates and saves the files,
kicks off the slow chunk+embed work as a background task via run_upload_job, and returns a job id right away; the
browser then polls the job's status. This used to be a single request streamed back over SSE, but that depends on
the browser's fetch()+ReadableStream reading behaving consistently, which isn't reliable across browsers in the same
way for every user's setup. Plain POST/GET with ordinary JSON bodies has no such dependency — it works the same
everywhere.
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

# Client-supplied filenames are untrusted, but the only thing that actually needs blocking is a path separator
# (which would escape the upload folder — already impossible here since Path(name).name below strips any directory
# component before this even runs) and characters that are outright invalid in a filename on some filesystem
# pAIring might run on. Everything else — spaces, parentheses, commas, accented or non-Latin letters (Hebrew,
# etc.) — is real, legitimate filename content and is left alone, so an uploaded file keeps a name that actually
# looks like what the user picked.
_UNSAFE_CHARS_RE = re.compile(r'[\x00-\x1f\x7f<>:"/\\|?*]')


def safe_filename(name: str) -> str:
    base = Path(name or "").name  # drops any directory component
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


# ---- Upload jobs -----------------------------------------------------------
#
# In-memory only, per-process — never shared or synced across this app's local instances (see
# app/services/instance_service.py). A poll landing on a different instance than the one that started the job
# would otherwise 404 mid-upload; start_upload_job below prefixes each job id with its own INSTANCE_INDEX
# specifically so app/services/instance_proxy.py's built-in load balancer can route a poll straight back to the
# right instance deterministically, without this dict itself needing to know anything about that. A job is small
# (a per-file status dict), so a plain dict keyed by job id is enough; _prune_old_jobs keeps it from growing
# forever across a long-running server's lifetime.

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
    """Same as get_job, but returns None (not just for a missing job) if `user` didn't start it and isn't an admin
    — a job id is an unguessable-in-practice UUID, but it's still exposed to the browser (network tab, logs), so
    nothing about it should double as an access token letting any authenticated user read another's upload
    filenames and per-file error text. The only caller (app/routers/documents.py) 404s either way regardless."""
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
    """The actual slow work (chunk + embed each accepted file), run as a fire-and-forget background task after the
    request that started it has already returned. Uses its own DB session — the request-scoped one from
    Depends(get_db) is closed by the time this runs."""
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
                    # sync_folder_stream re-checks this owner's *whole* folder, not just the files in this job —
                    # anything else (an older, already-up-to-date document) is real but not part of what this job
                    # promised to report on.
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
    """Validates every file against the admin-configured limits (size, supported type, per-user quota) and writes
    whatever's accepted to disk — the fast, synchronous half of an upload. Returns (accepted filenames, (original
    name, rejection reason) pairs, the target folder, the owner_id rows should be created under) for the caller to
    hand off to run_upload_job / start_upload_job below.

    A file that fails a limit check, isn't a supported type, or fails to parse (a corrupt/password-protected PDF,
    say) is reported as its own per-file error rather than failing the whole batch — the rest still go through.
    """
    limits = await get_rag_limits(db)
    max_file_bytes = limits.max_file_mb * 1024 * 1024
    quota_bytes = limits.max_user_space_mb * 1024 * 1024

    target_dir = GLOBAL_DIR if scope == "global" else user_folder(user.id)
    owner_id = None if scope == "global" else user.id

    # Tracks running usage as files are accepted one by one, so a batch can partially succeed (accept whatever
    # fits, reject the rest) instead of an all-or-nothing decision. Re-uploading a file that's already there only
    # counts its *new* size against the quota, not both old and new — its old size is subtracted out first.
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
    """Registers a new job and, if there's anything to actually process,
    schedules run_upload_job as a background task. Returns the job's
    initial status right away — see this module's docstring for why
    that's a snapshot to poll rather than a long-held request."""
    _prune_old_jobs()
    # Prefixed with this process's own instance index (see app/config.py) purely so
    # app/services/instance_proxy.py's built-in load balancer can route a later poll straight back to whichever
    # instance's in-memory _upload_jobs dict the job actually lives in, without any proxy-side state of its own —
    # this dict is never shared/synced across instances (see this module's own docstring on that).
    job_id = f"{INSTANCE_INDEX}-{uuid.uuid4().hex}"
    files_state = {name: {"status": "waiting", "chunk": 0, "chunks_total": None} for name in accepted}
    for filename, reason in rejects:
        files_state[filename] = {"status": "error", "error": reason}
    _upload_jobs[job_id] = {
        "created_at": time.time(),
        "done": not accepted,  # nothing left to do if every file was rejected up front
        "files": files_state,
        "result": None,
        # Who started this job — checked by app/routers/documents.py's GET /upload/{job_id} so one user can't
        # poll another's job status (filenames, per-file error text) just by guessing or observing its id.
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
