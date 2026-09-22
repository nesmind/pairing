"""Schemas for the ComfyUI process-supervision admin feature (Settings >
External servers) — see app/services/comfyui_service.py for the
supervisor that spawns/tracks/kills the actual subprocess these
describe, and app/services/comfyui_client.py / comfyui_pool.py for the
separate "talk to its HTTP API" half (local: app.config.COMFYUI_HOST;
remote: `remote_hosts` below, load-balanced) that this config has no
bearing on beyond picking which of those two applies."""

from pydantic import BaseModel, Field, model_validator

from app.schemas.common import MAX_REMOTE_HOSTS, ServerMode


class ComfyUIProcessConfig(BaseModel):
    """Admin-configured from Settings > External servers, saved even
    while unset/incomplete (see app.services.comfyui_service.start's own
    validation of the local-mode fields before actually spawning
    anything). `mode`/`remote_hosts` default in cleanly for a row saved
    before this feature existed — see settings_service.get_comfyui_config's
    own docstring."""

    mode: ServerMode = "local"
    remote_hosts: list[str] = Field(default_factory=list, max_length=MAX_REMOTE_HOSTS)
    # Local-mode only, ignored in "remote" mode:
    python_path: str | None = None
    main_py_path: str | None = None
    extra_args: str | None = None
    # Overrides for app.services.comfyui_installer's own install_stream —
    # see app.schemas.ollama_server_config.OllamaServerConfig's identical
    # pair for the full reasoning (None = use the built-in pinned default).
    install_repo: str | None = None
    install_version: str | None = None

    @model_validator(mode="after")
    def _fall_back_to_local_when_remote_has_no_hosts(self) -> "ComfyUIProcessConfig":
        """See app.schemas.ollama_server_config.OllamaServerConfig's
        identical validator for the full reasoning — "remote" with zero
        hosts is never a real, useful state, and the local-mode UI has
        no Save button while nothing's installed, so this is the only
        way back once that happens."""
        if self.mode == "remote" and not self.remote_hosts:
            self.mode = "local"
        return self


class ComfyUIStatus(BaseModel):
    """GET .../status response — `pid`/`healthy` are None when not
    currently running at all. `healthy` reflects a live HTTP ping to
    app.config.COMFYUI_HOST and is best-effort/informational only:
    `running` (the real PID check) is the actual source of truth.
    `installed` reflects app.services.comfyui_installer.is_installed()
    — independent of `running`."""

    running: bool
    installed: bool = False
    pid: int | None = None
    healthy: bool | None = None
