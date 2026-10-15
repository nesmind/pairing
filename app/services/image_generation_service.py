"""Text-to-image generation detached from the request via a background asyncio.Task. Like
reply_generation_service.py, the DB row is the source of truth (not document_upload.py's in-memory dict,
with its cross-instance gap): the ImageGenerationJob row is all a poller reads, correct on any instance.

Cancellation is DB-driven: the cancel endpoint flips the row to "cancelled" (any instance can), and the
running instance notices on its next poll, tells the engine to stop (stable-diffusion.cpp
/sdcpp/v1/jobs/{id}/cancel) and exits without writing a result.
"""

import asyncio
import logging
import os
import time

from sqlalchemy import update
from sqlalchemy.exc import InvalidRequestError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import IMAGES_DIR
from app.database import AsyncSessionLocal
from app.models import ImageGenerationJob
from app.schemas import ImageGenerationRequest
from app.services import image_engine_service, image_init, sdcpp_client, sdcpp_progress
from app.services.sdcpp_client import SdCppError

logger = logging.getLogger("llama_chat")

_POLL_INTERVAL_SECONDS = 2.0

# As reply_generation_service._background_generation_tasks: asyncio keeps only a weak reference to a bare
# create_task result, so the task could be garbage-collected mid-run.
_background_jobs: set[asyncio.Task] = set()


async def create_job(db: AsyncSession, owner_id: str, request: ImageGenerationRequest) -> ImageGenerationJob:
    """Inserts a "queued" row, schedules generation detached from the request and returns the job as it is now;
    the frontend polls GET.../jobs/{id}.
    """
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
        mode=request.mode,
        strength=request.strength if request.mode == "image_to_image" else None,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    if request.init_image:
        image_init.stash(job.id, request.init_image)
    _schedule(job.id)
    return job


def _schedule(job_id: str) -> asyncio.Task:
    task = asyncio.create_task(_run_job(job_id))
    _background_jobs.add(task)
    task.add_done_callback(_background_jobs.discard)
    return task


async def _run_job(job_id: str) -> None:
    """Runs on its own AsyncSessionLocal: the request's session may be gone by then."""
    async with AsyncSessionLocal() as db:
        job = await db.get(ImageGenerationJob, job_id)
        if job is None or job.status == "cancelled":
            image_init.discard(job_id)
            return

        engine = await image_engine_service.get_active_image_engine(db)
        job.status = "running"
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
            params.update(image_init.init_params(job, engine))
        except ValueError as exc:
            await _mark_error(db, job, str(exc))
            return
        try:
            image_bytes = await _sdcpp_image_bytes(db, job, params)
        except _JobCancelled:
            return
        except (SdCppError, _GenerationFailed) as exc:
            await _mark_error(db, job, str(exc))
            return
        if await _is_cancelled(db, job):
            return  # cancelled just as it finished: no result

        try:
            owner_dir = await image_engine_service.images_root(db, IMAGES_DIR) / job.owner_id
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
    """The user cancelled this job (its row says "cancelled"): stop quietly."""


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
    """Submits to stable-diffusion.cpp's native job API and polls it, honoring cancel. Raises SdCppError/
    _GenerationFailed/_JobCancelled. A cancel stops a still-queued engine job; one already generating finishes
    on builds without cancel_generating (the engine is shared, never restarted for this) and its result is discarded.
    """
    host, engine_job_id = await sdcpp_client.submit_job(params)
    job.engine_job_id = engine_job_id
    await db.commit()
    deadline = time.monotonic() + (await image_engine_service.get_sdcpp_config(db)).timeout_minutes * 60
    while time.monotonic() < deadline:
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


async def _mark_error(db: AsyncSession, job: ImageGenerationJob, message: str) -> None:
    job.status = "error"
    job.error_message = message
    await db.commit()


async def mark_interrupted_jobs_as_errored(db: AsyncSession) -> int:
    """Marks jobs still "running"/"queued" from before this process started as errors (an asyncio.Task can't
    survive a restart, else a row stays "generating" forever). Like
    conversation_service.mark_interrupted_messages_as_errored,
    called from startup_service.run_startup_tasks (primary only, single-instance restarts).
    """
    result = await db.execute(
        update(ImageGenerationJob)
        .where(ImageGenerationJob.status.in_(["queued", "running"]))
        .values(status="error", error_message="Interrupted by a server restart.")
    )
    await db.commit()
    return result.rowcount
