"""Image-generation JSON API: submit a prompt, poll a job, list your generations, fetch the file.
Lifecycle: app/services/image_generation_service.py; engine calls behind /checkpoints: comfyui_client.py,
sdcpp_client.py.
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import IMAGES_DIR
from app.database import get_db
from app.models import ImageGenerationJob, User
from app.schemas import CheckpointList, ImageGenerationJobOut, ImageGenerationRequest, OkResponse
from app.services import comfyui_client, image_engine_service, image_generation_service, sdcpp_client, sdcpp_progress
from app.services.auth_service import get_current_user
from app.services.comfyui_client import ComfyUIError
from app.services.sdcpp_client import SdCppError

router = APIRouter(prefix="/api/image-generation", tags=["image-generation"])


async def _get_own_job_or_404(db: AsyncSession, job_id: str, user: User) -> ImageGenerationJob:
    """ "Not yours" is a plain 404 (as conversation_service.get_accessible_conversation_or_404), so job ids can't be
    enumerated."""
    job = await db.get(ImageGenerationJob, job_id)
    if job is None or job.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Image generation job not found")
    return job


@router.post("/jobs", response_model=ImageGenerationJobOut)
async def create_job(
    body: ImageGenerationRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return await image_generation_service.create_job(db, user.id, body)


@router.get("/jobs", response_model=list[ImageGenerationJobOut])
async def list_jobs(db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    result = await db.execute(
        select(ImageGenerationJob)
        .where(ImageGenerationJob.owner_id == user.id)
        .order_by(ImageGenerationJob.created_at.desc())
        .limit(100)
    )
    return result.scalars().all()


@router.get("/jobs/{job_id}", response_model=ImageGenerationJobOut)
async def get_job(job_id: str, db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    job = await _get_own_job_or_404(db, job_id, user)
    return ImageGenerationJobOut.model_validate(job).model_copy(update=sdcpp_progress.for_job(job))


@router.post("/jobs/{job_id}/cancel", response_model=ImageGenerationJobOut)
async def cancel_job(job_id: str, db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    """Stops a queued or running job (the engine is told on the instance's next poll). Idempotent for finished jobs."""
    return await image_generation_service.cancel_job(db, await _get_own_job_or_404(db, job_id, user))


@router.get("/jobs/{job_id}/image")
async def get_job_image(job_id: str, db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    """The only way generated images are served (not under /static), gated by ownership like chat.py's
    get_attachment. Always PNG (written by this app).
    """
    job = await _get_own_job_or_404(db, job_id, user)
    if job.image_path is None:
        raise HTTPException(status_code=404, detail="This job has no generated image yet")
    path = await image_engine_service.find_image(db, IMAGES_DIR, job.image_path)
    if path is None:
        raise HTTPException(status_code=404, detail="The image file is missing (was the images folder changed?)")
    return FileResponse(path, filename=job.image_filename, media_type="image/png")


@router.delete("/jobs/{job_id}", response_model=OkResponse)
async def delete_job(job_id: str, db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    job = await _get_own_job_or_404(db, job_id, user)
    if job.image_path is not None:
        path = await image_engine_service.find_image(db, IMAGES_DIR, job.image_path)
        if path is not None:
            path.unlink(missing_ok=True)
    await db.delete(job)
    await db.commit()
    return OkResponse(ok=True)


@router.get("/checkpoints", response_model=CheckpointList)
async def list_checkpoints(db: AsyncSession = Depends(get_db), _user: User = Depends(get_current_user)):
    """Models of the active image engine."""
    engine = await image_engine_service.get_active_image_engine(db)
    try:
        if engine == "sdcpp":
            checkpoints = await sdcpp_client.list_checkpoints()
        else:
            checkpoints = await comfyui_client.list_checkpoints()
    except (ComfyUIError, SdCppError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return CheckpointList(checkpoints=checkpoints)
