"""Schemas for the ComfyUI process-supervision admin (Settings > External servers). Supervisor:
services/comfyui_service.py; HTTP API half: comfyui_client.py / comfyui_pool.py (local:
COMFYUI_HOST; remote: `remote_hosts`, load-balanced); this config only picks between them.
"""

from pydantic import BaseModel, Field, model_validator

from app.schemas.common import MAX_REMOTE_HOSTS, ServerMode


class ComfyUIProcessConfig(BaseModel):
    """Admin-configured; saved even when incomplete (comfyui_service.start validates local fields before spawning).
    `mode`/`remote_hosts` default cleanly for older rows (settings_service.get_comfyui_config).
    """

    mode: ServerMode = "local"
    remote_hosts: list[str] = Field(default_factory=list, max_length=MAX_REMOTE_HOSTS)
    # Local mode only:
    python_path: str | None = None
    main_py_path: str | None = None
    extra_args: str | None = None
    # Overrides for comfyui_installer.install_stream; see OllamaServerConfig's pair (None = built-in default).
    install_repo: str | None = None
    install_version: str | None = None

    @model_validator(mode="after")
    def _fall_back_to_local_when_remote_has_no_hosts(self) -> "ComfyUIProcessConfig":
        """Like OllamaServerConfig's validator: "remote" with no hosts is never useful, and the local UI has no
        Save button while nothing is installed, so this is the way back.
        """
        if self.mode == "remote" and not self.remote_hosts:
            self.mode = "local"
        return self


class ComfyUIStatus(BaseModel):
    """GET.../status. `pid`/`healthy` are None when not running. `healthy` is a best-effort ping of
    COMFYUI_HOST; `running` (PID check) is the truth. `installed` is comfyui_installer.is_installed.
    """

    running: bool
    installed: bool = False
    pid: int | None = None
    healthy: bool | None = None
