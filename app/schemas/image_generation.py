"""Request/response shapes for app/routers/image_generation.py — see
app/services/image_generation_service.py for the actual generation
lifecycle (queued/running/complete/error) these describe."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ImageGenerationRequest(BaseModel):
    """POST /api/image-generation/jobs body. Bounds are sanity limits on
    what a ComfyUI txt2img workflow can reasonably take, not tuned to any
    specific checkpoint — see app.services.comfyui_client's own workflow
    template for how these get applied."""

    prompt: str = Field(min_length=1, max_length=4000)
    negative_prompt: str | None = Field(default=None, max_length=4000)
    checkpoint: str = Field(min_length=1)
    width: int = Field(default=512, ge=64, le=2048)
    height: int = Field(default=512, ge=64, le=2048)
    steps: int = Field(default=1, ge=1, le=150)
    cfg: float = Field(default=1.0, ge=0.0, le=30.0)
    seed: int = Field(default=-1)


class ImageGenerationJobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    status: str
    prompt: str
    negative_prompt: str | None
    checkpoint: str
    width: int
    height: int
    steps: int
    cfg: float
    seed: int
    # Populated only once status="complete" — see
    # app.models.image_generation.ImageGenerationJob.url.
    url: str | None = None
    error_message: str | None
    created_at: datetime
    # Live progress while status="running" (stable-diffusion.cpp reports it from its own log): 0-100, what it's
    # doing, and a rough seconds-left. All None when the engine gives no progress — the page then shows an
    # indeterminate bar with an elapsed timer instead.
    progress: float | None = None
    stage: str | None = None
    eta_seconds: int | None = None


class CheckpointList(BaseModel):
    checkpoints: list[str]
