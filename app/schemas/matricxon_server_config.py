"""Schemas for the Matricxon process-supervision admin feature (Settings >
External servers) — see app/services/matricxon_process.py for the
supervisor that finds/starts/stops the actual sibling matricxon checkout
these describe (via its own scripts/start.sh|stop.sh, not a binary this
app launches directly — see project_dir's own docstring below), and
app/services/matricxon_pool.py / matricxon_client.py for the separate
"talk to its HTTP API" half. Mirrors app.schemas.ollama_server_config.
OllamaServerConfig's shape closely, with binary_path replaced by
project_dir — Matricxon has no standalone binary to point at (see
../matricxon/scripts/start.sh, a thin `uvicorn` launcher over a Python
venv, not a self-contained executable)."""

from pydantic import BaseModel, Field, model_validator

from app.schemas.common import MAX_REMOTE_HOSTS, ServerMode


class MatricxonServerConfig(BaseModel):
    """Admin-configured from Settings > External servers > Matricxon.
    Every local-mode field is optional — None means "use Matricxon's own
    built-in default" (see ../matricxon/app/config.py's Settings)."""

    mode: ServerMode = "local"
    remote_hosts: list[str] = Field(default_factory=list, max_length=MAX_REMOTE_HOSTS)
    # Path to the Matricxon project checkout (containing scripts/start.sh
    # etc) — the closest local-mode equivalent to OllamaServerConfig's
    # binary_path, but a directory, not an executable: see
    # app.services.matricxon_process._find_project_dir, which checks this
    # before falling back to the sibling "../matricxon" checkout this
    # app's own development layout already uses. None means "auto-detect".
    project_dir: str | None = None
    # Where Matricxon stores installed models — passed to scripts/start.sh as MATRICXON_MODELS_DIR
    # (see app.services.matricxon_process._build_env) and also where
    # app.services.matricxon_direct_puller writes a directly-pulled model in local mode. None
    # means "use Matricxon's own default inside the project directory above" (../matricxon/app/
    # config.py's Settings.models_dir, "./data/models" relative to its own cwd). Changing this
    # does not move already-downloaded models.
    models_path: str | None = None
    # Local-mode only, ignored in "remote" mode — passed to matricxon's own scripts/start.sh as
    # MATRICXON_MAX_LOADED_MODELS (see app.services.matricxon_process._build_env). keep_alive_seconds/device were
    # dropped from this form (not from matricxon's own MATRICXON_DEFAULT_KEEP_ALIVE_SECONDS/MATRICXON_DEVICE env
    # vars, which still work if set on the process directly) — device has nothing but "cpu" to meaningfully choose
    # between today (matricxon has no GPU support yet), and keep_alive tuning wasn't worth the extra form field.
    max_loaded_models: int | None = None
    # Local-mode only — passed as MATRICXON_MEMORY_SAFETY_MARGIN. None means "use matricxon's own built-in
    # default" (Settings.memory_safety_margin, 1.5x — see ../matricxon/app/config.py). Bounded to the same
    # 1.1-1.8 range matricxon's own Settings field enforces server-side (below 1.1 stops meaning anything as a
    # *safety* margin; above 1.8 has no evidence behind it and risks the exact OOM-kill the check exists to
    # avoid) — validated here too so a bad value is rejected at Save time with a clear message instead of
    # silently failing matricxon's own env-var validation on the next restart.
    memory_safety_margin: float | None = Field(default=None, ge=1.1, le=1.8)
    # Local-mode only — passed as MATRICXON_ENABLE_QUANTIZED_NATIVE_COMPUTE. Mirrors matricxon's own
    # Settings.enable_quantized_native_compute (../matricxon/app/config.py) default of False — see that
    # field's own docstring for what it trades off (real quantized-native compute, never materializing a full
    # dequantized weight, at the cost of slower decode).
    enable_quantized_native_compute: bool = False
    # Local-mode only — passed as MATRICXON_LOG_LEVEL. Mirrors matricxon's own Settings.log_level (same file):
    # 0 = warnings/errors only, 1 = coarse per-request pipeline milestones, 2 = adds a per-decoder-layer trace.
    log_level: int = Field(default=0, ge=0, le=2)
    # Overrides for app.services.matricxon_installer's own install_stream
    # — kept for shape-parity with OllamaServerConfig/ComfyUIProcessConfig,
    # but not yet functional: Matricxon isn't published on GitHub yet (see
    # matricxon_installer's own docstring), so these fields have nothing
    # real to point at until a release exists.
    install_repo: str | None = None
    install_version: str | None = None

    @model_validator(mode="after")
    def _fall_back_to_local_when_remote_has_no_hosts(self) -> "MatricxonServerConfig":
        """See OllamaServerConfig's identical validator for the full reasoning."""
        if self.mode == "remote" and not self.remote_hosts:
            self.mode = "local"
        return self


class MatricxonAutoDetectedPath(BaseModel):
    """What app.services.matricxon_process._find_project_dir resolves to
    right now with no admin override — Settings > External servers shows
    this as the project_dir field's own placeholder, mirroring
    OllamaAutoDetectedPath's identical purpose for binary_path."""

    path: str | None = None
    # What app.services.matricxon_process.auto_detect_models_dir() resolves to — the models_path
    # field's own placeholder, same "show what blank actually resolves to" idea as `path` above.
    models_path: str | None = None


class MatricxonServerStatus(BaseModel):
    """GET .../status response — mirrors OllamaServerStatus's exact
    shape. `pid` is read from Matricxon's own run/matricxon.pid (written
    by scripts/start.sh) rather than tracked independently."""

    running: bool
    installed: bool = False
    pid: int | None = None
    healthy: bool | None = None
