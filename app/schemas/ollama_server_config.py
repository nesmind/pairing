"""Schemas for the Ollama process-supervision admin (Settings > External servers). Supervisor:
services/ollama_process.py (`ollama serve`); HTTP half: ollama_pool.py (local: managed process; remote:
`remote_hosts`, load-balanced), which this config only chooses between. Replaces the old
OllamaConcurrency schemas: num_parallel is now one of several local-mode params.
"""

from pydantic import BaseModel, Field, model_validator

from app.schemas.common import MAX_REMOTE_HOSTS, ServerMode


class OllamaServerConfig(BaseModel):
    """Admin-configured. Local-mode fields are optional: None = let Ollama use its default. Explicit fields
    (not a free-text arg string like ComfyUI's `extra_args`) since Ollama's env vars are a small, known set.
    """

    mode: ServerMode = "local"
    remote_hosts: list[str] = Field(default_factory=list, max_length=MAX_REMOTE_HOSTS)
    # For an Ollama installed in a non-standard place instead of "Install from GitHub"; ollama_process._find_binary
    # checks it before PATH and fixed locations. None = auto-detect. One field (not ComfyUI's python/main pair)
    # since the binary is self-contained.
    binary_path: str | None = None
    # Where pulled models go (OLLAMA_MODELS). None = pAIring's own models/ folder (ollama_process._build_env),
    # not ~/.ollama/models, to avoid colliding with a system-wide Ollama. Changing it moves nothing.
    models_path: str | None = None
    # Local mode only, ignored in "remote": passed to `ollama serve` as the matching OLLAMA_* env var
    # (ollama_process.apply_local_config):
    num_parallel: int | None = None
    keep_alive: str | None = None  # OLLAMA_KEEP_ALIVE syntax, e.g. "5m", "24h"
    max_loaded_models: int | None = None
    context_length: int | None = None
    # Overrides for ollama_installer.install_stream. None = app.config.OLLAMA_GITHUB_REPO/OLLAMA_PINNED_VERSION;
    # set to pin another fork or release (Settings > External servers' "Use a different version or repo").
    install_repo: str | None = None
    install_version: str | None = None

    @model_validator(mode="after")
    def _fall_back_to_local_when_remote_has_no_hosts(self) -> "OllamaServerConfig":
        """ "Remote" with no hosts is never useful, and local mode has no Save button before anything is installed, so
        an admin could get stuck. Normalizing here also self-heals an already-stuck row on read.
        """
        if self.mode == "remote" and not self.remote_hosts:
            self.mode = "local"
        return self


class OllamaAutoDetectedPath(BaseModel):
    """What ollama_process._find_binary resolves to with no override (PATH, then fixed locations); the
    binary_path placeholder.
    """

    path: str | None = None
    # What ollama_process.auto_detect_models_path resolves to (always a concrete path); the models_path placeholder.
    models_path: str | None = None


class OllamaServerStatus(BaseModel):
    """GET.../status, same shape as ComfyUIStatus. `pid`/`healthy` are None when not running. `installed`
    (ollama_process.is_installed) is independent of `running`: the binary may be present but stopped.
    """

    running: bool
    installed: bool = False
    pid: int | None = None
    healthy: bool | None = None
