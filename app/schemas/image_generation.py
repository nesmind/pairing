"""Request/response shapes for app/routers/image_generation.py — see
app/services/image_generation_service.py for the actual generation
lifecycle (queued/running/complete/error) these describe."""

import base64
import binascii
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

GenerationMode = Literal["text_to_image", "image_to_image"]
_MAX_INIT_IMAGE_B64 = 20_000_000
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) from a PNG's IHDR, or None if `data` isn't a PNG."""
    if len(data) < 24 or data[:8] != _PNG_SIGNATURE or data[12:16] != b"IHDR":
        return None
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


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
    mode: GenerationMode = "text_to_image"
    # image_to_image only: a base64 PNG already sized width x height (the page resizes it), and how far the result
    # may move away from it (0-1; near 0 keeps the picture, 1 nearly ignores it).
    init_image: str | None = Field(default=None, max_length=_MAX_INIT_IMAGE_B64)
    strength: float = Field(default=0.75, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_source_image(self) -> "ImageGenerationRequest":
        if self.mode == "text_to_image":
            self.init_image = None
            return self
        if not self.init_image:
            raise ValueError("Image-to-image needs a source image.")
        try:
            size = _png_size(base64.b64decode(self.init_image, validate=True))
        except (binascii.Error, ValueError) as exc:
            raise ValueError("The source image is not valid base64.") from exc
        if size is None:
            raise ValueError("The source image must be a PNG.")
        if size != (self.width, self.height):
            raise ValueError(f"The source image is {size[0]}x{size[1]} but the request is {self.width}x{self.height}.")
        return self


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
    mode: str = "text_to_image"
    strength: float | None = None
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
