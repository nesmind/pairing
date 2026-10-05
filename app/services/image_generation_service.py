"""
Text-to-image generation, detached from the initiating request via a
background asyncio.Task — mirrors app/services/reply_generation_service.py's
DB-row-as-source-of-truth shape (not app/services/document_upload.py's
in-memory-dict approach, which has a real documented cross-instance gap
this doesn't need to inherit): the ImageGenerationJob row itself is the
only state a poller ever needs to read, so it's correct regardless of
which instance later serves a poll.

Cancellation is DB-driven too: the cancel endpoint just flips the row to "cancelled" (so any instance can do it),
and the instance running the job notices on its next poll, tells the engine to stop (stable-diffusion.cpp
/sdcpp/v1/jobs/{id}/cancel, ComfyUI /interrupt) and exits without writing a result.
"""

import asyncio
import logging
import os

from sqlalchemy import update
from sqlalchemy.exc import InvalidRequestError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import IMAGES_DIR
from app.database import AsyncSessionLocal
from app.models import ImageGenerationJob
from app.schemas import ImageGenerationRequest
from app.services import comfyui_client, image_engine_service, sdcpp_client, sdcpp_progress
from app.services.comfyui_client import ComfyUIError
from app.services.sdcpp_client import SdCppError

logger = logging.getLogger("llama_chat")

_POLL_INTERVAL_SECONDS = 2.0
# Bounded so a permanently-stuck engine job doesn't leave a job "running" forever: ~5 minutes for ComfyUI,
# ~30 for stable-diffusion.cpp (a CPU-only run can legitimately take many minutes).
_MAX_POLL_ATTEMPTS = 150
_SDCPP_MAX_POLL_ATTEMPTS = 900

# Same reasoning as reply_generation_service._background_generation_tasks:
# asyncio only holds a *weak* reference to a bare asyncio.create_task()
# result, so without this the task can be garbage-collected mid-run.
_background_jobs: set[asyncio.Task] = set()


async def create_job(db: AsyncSession, owner_id: str, request: ImageGenerationRequest) -> ImageGenerationJob:
    """Inserts a "queued" row and schedules its generation fully
    detached from this request, then returns immediately — the caller's
    own response is the job as it exists right now, not the finished
    result; the frontend polls GET .../jobs/{id} for progress."""
    seed = request.seed if request.seed >= 0 else int.from_bytes(os.urandom(4), "big")
    job = ImageGenerationJob(
        owner_id=owner_id,
        status="queued",
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        checkpoint=request.checkpoint,
        width=request.width,
        height=request.height,
        steps=request.steps,
        cfg=request.cfg,
        seed=seed,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    _schedule(job.id)
    return job


def _schedule(job_id: str) -> asyncio.Task:
    task = asyncio.create_task(_run_job(job_id))
    _background_jobs.add(task)
    task.add_done_callback(_background_jobs.discard)
    return task


async def _run_job(job_id: str) -> None:
    """Runs against its own fresh AsyncSessionLocal() — the request's db
    session may already be gone by the time this finishes."""
    async with AsyncSessionLocal() as db:
        job = await db.get(ImageGenerationJob, job_id)
        if job is None or job.status == "cancelled":
            return  # deleted, or cancelled while still queued

        engine = await image_engine_service.get_active_image_engine(db)
        job.status = "running"
        if engine == "sdcpp":
            job.log_offset = sdcpp_progress.current_offset()
        await db.commit()

        params = {
            "checkpoint": job.checkpoint,
            "prompt": job.prompt,
            "negative_prompt": job.negative_prompt,
            "width": job.width,
            "height": job.height,
            "steps": job.steps,
            "cfg": job.cfg,
            "seed": job.seed,
        }
        try:
            if engine == "sdcpp":
                image_bytes = await _sdcpp_image_bytes(db, job, params)
            else:
                image_bytes = await _comfyui_image_bytes(db, job, params)
        except _JobCancelled:
            return  # the row already says "cancelled"
        except (SdCppError, ComfyUIError, _GenerationFailed) as exc:
            await _mark_error(db, job, str(exc))
            return
        if await _is_cancelled(db, job):
            return  # cancelled just as it finished — no result

        try:
            owner_dir = IMAGES_DIR / job.owner_id
            owner_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{job.id}.png"
            (owner_dir / filename).write_bytes(image_bytes)
        except OSError:
            logger.exception("Failed to save generated image for job %s", job_id)
            await _mark_error(db, job, "Failed to save the generated image.")
            return

        job.image_path = f"{job.owner_id}/{filename}"
        job.image_filename = filename
        job.status = "complete"
        await db.commit()


class _JobCancelled(Exception):
    """The user cancelled this job (its row says "cancelled") — stop quietly."""


async def _is_cancelled(db: AsyncSession, job: ImageGenerationJob) -> bool:
    """Re-reads the row (a cancel comes from another request, possibly another instance). A deleted row counts."""
    try:
        await db.refresh(job)
    except InvalidRequestError:
        return True
    return job.status == "cancelled"


async def cancel_job(db: AsyncSession, job: ImageGenerationJob) -> ImageGenerationJob:
    """Marks a queued/running job cancelled; the running instance stops the engine on its next poll. A job that
    already finished is returned unchanged."""
    if job.status in ("queued", "running"):
        job.status = "cancelled"
        await db.commit()
    return job


class _GenerationFailed(Exception):
    """A failure with a user-facing message that isn't a client-level error (e.g. a timeout)."""


async def _sdcpp_image_bytes(db: AsyncSession, job: ImageGenerationJob, params: dict) -> bytes:
    """Submit to stable-diffusion.cpp's native job API and poll it, honoring a cancel. Raises SdCppError/
    _GenerationFailed/_JobCancelled. A cancel stops a still-queued engine job; one already generating runs to
    completion on builds without cancel_generating (the engine is shared, so it's never restarted for this) — the
    result is simply discarded."""
    host, engine_job_id = await sdcpp_client.submit_job(params)
    job.comfy_prompt_id = engine_job_id  # the engine's own job id, whichever engine
    await db.commit()
    for _ in range(_SDCPP_MAX_POLL_ATTEMPTS):
        if await _is_cancelled(db, job):
            await sdcpp_client.cancel_job(host, engine_job_id)
            raise _JobCancelled
        state = await sdcpp_client.get_job(host, engine_job_id)
        status = state.get("status")
        if status == "completed":
            return sdcpp_client.image_from_job(state)
        if status in ("failed", "cancelled"):
            raise _GenerationFailed((state.get("error") or {}).get("message") or f"Generation {status}.")
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)
    await sdcpp_client.cancel_job(host, engine_job_id)
    raise _GenerationFailed("Generation timed out.")


async def _comfyui_image_bytes(db: AsyncSession, job: ImageGenerationJob, params: dict) -> bytes:
    """Submit to ComfyUI, poll its /history, fetch the PNG. Raises ComfyUIError/_GenerationFailed."""
    host, prompt_id = await comfyui_client.submit_job(params)
    job.comfy_prompt_id = prompt_id
    await db.commit()

    history = await _poll_until_done(db, job, host, prompt_id)
    if history is None:
        raise _GenerationFailed("Generation timed out.")
    try:
        image_entry = history["outputs"][comfyui_client.SAVE_IMAGE_NODE_ID]["images"][0]
        return await comfyui_client.fetch_image_bytes(
            host, image_entry["filename"], image_entry.get("subfolder", ""), image_entry.get("type", "output")
        )
    except (ComfyUIError, KeyError, IndexError) as exc:
        raise _GenerationFailed(f"Could not retrieve the generated image: {exc}") from exc


async def _poll_until_done(db: AsyncSession, job: ImageGenerationJob, host: str, prompt_id: str) -> dict | None:
    """Bounded polling of GET /history/{id} against the exact host the job
    was submitted to (see comfyui_client's own docstring on why this can't
    fail over to a different host) — no websocket live progress in this
    first version, plain polling is enough. Returns None if
    _MAX_POLL_ATTEMPTS is exhausted without ComfyUI ever finishing; raises _JobCancelled if the user cancels."""
    for _ in range(_MAX_POLL_ATTEMPTS):
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        if await _is_cancelled(db, job):
            await comfyui_client.cancel_job(host, prompt_id)
            raise _JobCancelled
        history = await comfyui_client.get_history(host, prompt_id)
        if history is not None:
            return history
    return None


async def _mark_error(db: AsyncSession, job: ImageGenerationJob, message: str) -> None:
    job.status = "error"
    job.error_message = message
    await db.commit()


def progress_for(job: ImageGenerationJob) -> dict:
    """ImageGenerationJobOut's progress/stage/eta fields for a running stable-diffusion.cpp job, else {}."""
    if job.status != "running" or job.log_offset is None:
        return {}
    found = sdcpp_progress.read_progress(job.log_offset)
    if found is None:
        return {}
    return {"progress": round(found.progress, 1), "stage": found.stage, "eta_seconds": found.eta_seconds}


async def mark_interrupted_jobs_as_errored(db: AsyncSession) -> int:
    """Sweeps any job still marked "running"/"queued" from before this
    process started — an asyncio.Task can't survive a restart, so without
    this a row would stay stuck showing a permanent "generating" state.
    Same treatment conversation_service.mark_interrupted_messages_as_errored
    gives a stuck "streaming" Message; called from
    app.services.startup_service.run_startup_tasks the same way,
    primary-only, single-instance-restart scope only."""
    result = await db.execute(
        update(ImageGenerationJob)
        .where(ImageGenerationJob.status.in_(["queued", "running"]))
        .values(status="error", error_message="Interrupted by a server restart.")
    )
    await db.commit()
    return result.rowcount
