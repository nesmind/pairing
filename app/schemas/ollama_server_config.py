"""Schemas for the Ollama process-supervision admin feature (Settings >
External servers) — see app/services/ollama_process.py for the
supervisor that finds/starts/stops the actual `ollama serve` process
these describe, and app/services/ollama_pool.py for the separate "talk
to its HTTP API" half (local: this app's own managed process; remote:
`remote_hosts` below, load-balanced) that this config has no bearing on
beyond picking which of those two applies. Replaces the old, narrower
OllamaConcurrency/OllamaConcurrencyUpdate schemas — num_parallel is now
one of several local-mode parameters here instead of its own standalone
settings page."""

from pydantic import BaseModel, Field, model_validator

from app.schemas.common import MAX_REMOTE_HOSTS, ServerMode


class OllamaServerConfig(BaseModel):
    """Admin-configured from Settings > External servers. Every
    local-mode field is optional — None means "don't set it, let Ollama
    use its own built-in default" — modeled as explicit fields (not a
    free-text arg string like ComfyUI's `extra_args`) since Ollama's own
    tunable env vars are a small, well-known, finite set."""

    mode: ServerMode = "local"
    remote_hosts: list[str] = Field(default_factory=list, max_length=MAX_REMOTE_HOSTS)
    # An admin pointing this app at an Ollama they already have installed
    # somewhere non-standard, instead of using the "Install from GitHub"
    # button — see app.services.ollama_process._find_binary, which checks
    # this before falling back to PATH/the usual fixed locations. None
    # means "auto-detect as before" — mirrors ComfyUIProcessConfig's
    # identical python_path/main_py_path pair, just collapsed to one
    # field since Ollama's binary is self-contained (no separate
    # interpreter + entry-point split the way a Python app needs).
    binary_path: str | None = None
    # Where Ollama stores pulled models (OLLAMA_MODELS) — None means "use pAIring's own
    # self-contained models/ folder" (app.services.ollama_process._build_env's own default),
    # not Ollama's usual ~/.ollama/models: that isolation is deliberate (see this field's own
    # discussion — keeping models scoped to this app avoids colliding with an unrelated
    # system-wide Ollama install). Changing this does not move already-downloaded models.
    models_path: str | None = None
    # Local-mode only, ignored in "remote" mode — passed to `ollama
    # serve` as the matching OLLAMA_* env var (see
    # app.services.ollama_process.apply_local_config):
    num_parallel: int | None = None
    keep_alive: str | None = None  # Ollama's own OLLAMA_KEEP_ALIVE syntax, e.g. "5m", "24h"
    max_loaded_models: int | None = None
    context_length: int | None = None
    # Overrides for app.services.ollama_installer's own install_stream —
    # None means "use app.config.OLLAMA_GITHUB_REPO/OLLAMA_PINNED_VERSION",
    # the maintainer-picked default; set to pin a different fork/release
    # instead (see Settings > External servers' "Use a different version
    # or repo" link).
    install_repo: str | None = None
    install_version: str | None = None

    @model_validator(mode="after")
    def _fall_back_to_local_when_remote_has_no_hosts(self) -> "OllamaServerConfig":
        """ "Remote" with zero hosts isn't a real, useful state — and
        before local mode has anything installed, its own UI has no Save
        button (nothing to configure yet), so a stray remote-with-no-
        hosts save would otherwise leave an admin stuck with no way back
        to local. Normalizing here, not just in the UI, means an already-
        stuck row also self-heals the next time it's read back."""
        if self.mode == "remote" and not self.remote_hosts:
            self.mode = "local"
        return self


class OllamaAutoDetectedPath(BaseModel):
    """What app.services.ollama_process._find_binary resolves to right
    now with no admin override (PATH, then the usual fixed fallback
    locations) — Settings > External servers shows this as the
    binary_path field's own placeholder, so "leave this blank" has a
    concrete, visible value instead of just the words "auto-detected
    from PATH"."""

    path: str | None = None
    # What app.services.ollama_process.auto_detect_models_path() resolves to (always a concrete
    # path — unlike `path` above, there's no "not found" case for this one) — the models_path
    # field's own placeholder, same "show what blank actually resolves to" idea.
    models_path: str | None = None


class OllamaServerStatus(BaseModel):
    """GET .../status response — mirrors ComfyUIStatus's exact shape.
    `pid`/`healthy` are None when not currently running at all.
    `installed` reflects app.services.ollama_process.is_installed() —
    independent of `running`, since a binary can be present but stopped,
    or (before Settings > External servers' "Install" button is used)
    absent entirely."""

    running: bool
    installed: bool = False
    pid: int | None = None
    healthy: bool | None = None
