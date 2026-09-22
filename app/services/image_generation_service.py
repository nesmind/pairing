"""
Text-to-image generation, detached from the initiating request via a
background asyncio.Task — mirrors app/services/reply_generation_service.py's
DB-row-as-source-of-truth shape (not app/services/document_upload.py's
in-memory-dict approach, which has a real documented cross-instance gap
this doesn't need to inherit): the ImageGenerationJob row itself is the
only state a poller ever needs to read, so it's correct regardless of
which instance later serves a poll.

No cancellation/cross-instance-watchdog machinery like a chat reply gets
(compare reply_cancellation_service/reply_cross_instance_service) — a
generation job has no live SSE viewer to disconnect from and no delete-
while-running UX in this first version, so that complexity is genuinely
not needed here.
"""

import asyncio
import logging
import os

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import IMAGES_DIR
from app.database import AsyncSessionLocal
from app.models import ImageGenerationJob
from app.schemas import ImageGenerationRequest
from app.services import comfyui_client
from app.services.comfyui_client import ComfyUIError

logger = logging.getLogger("llama_chat")

_POLL_INTERVAL_SECONDS = 2.0
# ~5 minutes total — generous for CPU-bound generation, bounded so a
# permanently-stuck ComfyUI job doesn't leave a job "running" forever.
_MAX_POLL_ATTEMPTS = 150

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
        if job is None:
            return  # deleted out from under this generation

        job.status = "running"
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
            host, prompt_id = await comfyui_client.submit_job(params)
        except ComfyUIError as exc:
            await _mark_error(db, job, str(exc))
            return
        job.comfy_prompt_id = prompt_id
        await db.commit()

        try:
            history = await _poll_until_done(host, prompt_id)
        except ComfyUIError as exc:
            await _mark_error(db, job, str(exc))
            return
        if history is None:
            await _mark_error(db, job, "Generation timed out.")
            return

        try:
            image_entry = history["outputs"][comfyui_client.SAVE_IMAGE_NODE_ID]["images"][0]
            image_bytes = await comfyui_client.fetch_image_bytes(
                host, image_entry["filename"], image_entry.get("subfolder", ""), image_entry.get("type", "output")
            )
        except (ComfyUIError, KeyError, IndexError) as exc:
            await _mark_error(db, job, f"Could not retrieve the generated image: {exc}")
            return

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


async def _poll_until_done(host: str, prompt_id: str) -> dict | None:
    """Bounded polling of GET /history/{id} against the exact host the job
    was submitted to (see comfyui_client's own docstring on why this can't
    fail over to a different host) — no websocket live progress in this
    first version, plain polling is enough. Returns None if
    _MAX_POLL_ATTEMPTS is exhausted without ComfyUI ever finishing."""
    for _ in range(_MAX_POLL_ATTEMPTS):
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)
        history = await comfyui_client.get_history(host, prompt_id)
        if history is not None:
            return history
    return None


async def _mark_error(db: AsyncSession, job: ImageGenerationJob, message: str) -> None:
    job.status = "error"
    job.error_message = message
    await db.commit()


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
