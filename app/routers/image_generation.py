"""
The image-generation JSON API: submit a prompt, poll a job's status,
list your own past generations, and fetch the resulting file. See
app/services/image_generation_service.py for the actual generation
lifecycle these wrap, and app/services/comfyui_client.py for the ComfyUI
HTTP calls behind /checkpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import IMAGES_DIR
from app.database import get_db
from app.models import ImageGenerationJob, User
from app.schemas import CheckpointList, ImageGenerationJobOut, ImageGenerationRequest, OkResponse
from app.services import comfyui_client, image_generation_service
from app.services.auth_service import get_current_user
from app.services.comfyui_client import ComfyUIError

router = APIRouter(prefix="/api/image-generation", tags=["image-generation"])


async def _get_own_job_or_404(db: AsyncSession, job_id: str, user: User) -> ImageGenerationJob:
    """Folds "exists but not yours" into a plain 404 — same convention
    conversation_service.get_accessible_conversation_or_404 uses, so a
    caller can't distinguish "not found" from "not yours" and enumerate
    other users' job ids."""
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
    return await _get_own_job_or_404(db, job_id, user)


@router.get("/jobs/{job_id}/image")
async def get_job_image(job_id: str, db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    """The only way a generated image is ever served — not mounted under
    /static, so access stays gated by ownership, same reasoning as
    app/routers/chat.py's get_attachment. Always PNG: ComfyUI's own
    SaveImage node (see comfyui_client's workflow template) writes PNGs,
    and this app wrote the file itself — no guessing needed."""
    job = await _get_own_job_or_404(db, job_id, user)
    if job.image_path is None:
        raise HTTPException(status_code=404, detail="This job has no generated image yet")
    return FileResponse(IMAGES_DIR / job.image_path, filename=job.image_filename, media_type="image/png")


@router.delete("/jobs/{job_id}", response_model=OkResponse)
async def delete_job(job_id: str, db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    job = await _get_own_job_or_404(db, job_id, user)
    if job.image_path is not None:
        (IMAGES_DIR / job.image_path).unlink(missing_ok=True)
    await db.delete(job)
    await db.commit()
    return OkResponse(ok=True)


@router.get("/checkpoints", response_model=CheckpointList)
async def list_checkpoints(_user: User = Depends(get_current_user)):
    try:
        checkpoints = await comfyui_client.list_checkpoints()
    except ComfyUIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return CheckpointList(checkpoints=checkpoints)
