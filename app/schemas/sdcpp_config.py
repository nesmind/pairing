"""Schemas for the stable-diffusion.cpp image engine (Settings > Image). Supervisor: services/sdcpp_service.py
(`sd-server`); HTTP half: sdcpp_client.py / sdcpp_pool.py (local: app.config.SDCPP_HOST; remote:
`remote_hosts`).
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas.common import MAX_REMOTE_HOSTS, ServerMode

ImageEngineName = Literal["sdcpp"]
# Which prebuilt release to fetch ("auto" picks Vulkan with a GPU + Vulkan loader), or "source": build with
# CMake (Linux; sdcpp_source_build).
SdCppBuild = Literal["auto", "cpu", "vulkan", "rocm", "source"]
DEFAULT_IMAGE_ENGINE: ImageEngineName = "sdcpp"


class SdCppConfig(BaseModel):
    """Admin-configured; saved even when incomplete (sdcpp_service.start validates)."""

    mode: ServerMode = "local"
    remote_hosts: list[str] = Field(default_factory=list, max_length=MAX_REMOTE_HOSTS)
    # Local mode only:
    binary_path: str | None = None
    model_path: str | None = None  # .gguf/.safetensors/.ckpt loaded at launch
    extra_args: str | None = None
    # Where image models are stored; None = auto (a `diffusion` sibling of the Matricxon/Ollama models folder).
    models_path: str | None = None
    # Root of saved images (<root>/<user id>/<job id>.png), either mode; None = IMAGES_DIR.
    images_path: str | None = None
    install_repo: str | None = None
    install_version: str | None = None
    build: SdCppBuild = "auto"
    # Max time per image before the job is cancelled as timed out, either mode.
    timeout_minutes: int = Field(30, ge=1, le=1440)

    @model_validator(mode="after")
    def _fall_back_to_local_when_remote_has_no_hosts(self) -> "SdCppConfig":
        """Like OllamaServerConfig's validator: "remote" with no hosts is never useful, and the local UI has no Save
        button while nothing is installed, so this is the way back."""
        if self.mode == "remote" and not self.remote_hosts:
            self.mode = "local"
        return self


class SdCppStatus(BaseModel):
    """GET.../status: `healthy` is a best-effort ping; `running` (PID check) is the truth."""

    running: bool
    installed: bool = False
    pid: int | None = None
    healthy: bool | None = None
    # Build tools missing for "Build from source" (empty = usable).
    source_build_missing: list[str] = Field(default_factory=list)


class ImageEngineConfig(BaseModel):
    """Which engine serves the Images page."""

    active_image_engine: ImageEngineName = DEFAULT_IMAGE_ENGINE
