"""Schemas for the stable-diffusion.cpp image engine (Settings > Image) — see
app/services/sdcpp_service.py for the supervisor that spawns/tracks `sd-server`, and
app/services/sdcpp_client.py / sdcpp_pool.py for the HTTP half (local: app.config.SDCPP_HOST; remote:
`remote_hosts`). Mirrors app.schemas.comfyui_config."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas.common import MAX_REMOTE_HOSTS, ServerMode

ImageEngineName = Literal["sdcpp", "comfyui"]
# Which prebuilt release the installer fetches; "auto" picks Vulkan when a GPU + Vulkan loader are present.
SdCppBuild = Literal["auto", "cpu", "vulkan", "rocm"]
DEFAULT_IMAGE_ENGINE: ImageEngineName = "sdcpp"


class SdCppConfig(BaseModel):
    """Admin-configured; saved even while incomplete (see sdcpp_service.start's own validation)."""

    mode: ServerMode = "local"
    remote_hosts: list[str] = Field(default_factory=list, max_length=MAX_REMOTE_HOSTS)
    # Local-mode only:
    binary_path: str | None = None  # the sd-server executable
    model_path: str | None = None  # .gguf/.safetensors/.ckpt loaded at launch
    extra_args: str | None = None
    # Where image models are stored; None = auto (a `diffusion` sibling of the Matricxon/Ollama models folder).
    models_path: str | None = None
    # Root of the saved images (<root>/<user id>/<job id>.png), whichever mode; None = the IMAGES_DIR default.
    images_path: str | None = None
    install_repo: str | None = None
    install_version: str | None = None
    build: SdCppBuild = "auto"

    @model_validator(mode="after")
    def _fall_back_to_local_when_remote_has_no_hosts(self) -> "SdCppConfig":
        """Same reasoning as ComfyUIProcessConfig's identical validator."""
        if self.mode == "remote" and not self.remote_hosts:
            self.mode = "local"
        return self


class SdCppStatus(BaseModel):
    """GET .../status — `healthy` is a best-effort ping, `running` (the PID check) is the truth."""

    running: bool
    installed: bool = False
    pid: int | None = None
    healthy: bool | None = None


class ImageEngineConfig(BaseModel):
    """Which engine serves the Images page right now."""

    active_image_engine: ImageEngineName = DEFAULT_IMAGE_ENGINE
