"""Schemas for the Matricxon process-supervision admin (Settings > External servers). Supervisor:
services/matricxon_process.py (runs the sibling checkout's scripts/start.sh|stop.sh, no binary);
HTTP half: matricxon_pool.py / matricxon_client.py. Mirrors OllamaServerConfig with project_dir
instead of binary_path (Matricxon is a Python venv launched by a `uvicorn` script).
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas.common import MAX_REMOTE_HOSTS, ServerMode


class MatricxonServerConfig(BaseModel):
    """Admin-configured. Local-mode fields are optional: None = Matricxon's own default (../matricxon/app/config.py
    Settings)."""

    mode: ServerMode = "local"
    remote_hosts: list[str] = Field(default_factory=list, max_length=MAX_REMOTE_HOSTS)
    # Matricxon checkout (with scripts/start.sh): a directory, unlike Ollama's binary_path.
    # matricxon_process._find_project_dir checks it before the sibling "../matricxon". None = auto-detect.
    project_dir: str | None = None
    # Where installed models are stored: MATRICXON_MODELS_DIR for scripts/start.sh and where
    # matricxon_direct_puller writes in local mode. None = Matricxon's default ("./data/models" in its project).
    # Changing it moves nothing.
    models_path: str | None = None
    # Local mode only: MATRICXON_MAX_LOADED_MODELS (matricxon_process._build_env).
    max_loaded_models: int | None = None
    # Local mode only: minutes an idle model stays in RAM (MATRICXON_DEFAULT_KEEP_ALIVE_SECONDS = minutes * 60;
    # None = 5 minutes). Unloading drops the chat's prompt cache, so the next message re-processes the whole
    # conversation; longer avoids that at the cost of RAM. 1 minute to 24 hours.
    keep_alive_minutes: int | None = Field(default=None, ge=1, le=1440)
    # Local mode only, concurrent-user tuning (None = Matricxon's default): conversation caches per loaded
    # model (MATRICXON_PROMPT_CACHE_SLOTS, 4), memory for them in MB (MATRICXON_PROMPT_CACHE_BUDGET_MB, 2048;
    # the first limit hit evicts the least recently used), and concurrent replies per model with batched
    # decode steps (MATRICXON_MAX_DECODE_BATCH, 8; others queue).
    prompt_cache_slots: int | None = Field(default=None, ge=1, le=64)
    prompt_cache_budget_mb: int | None = Field(default=None, ge=64, le=1_048_576)
    max_decode_batch: int | None = Field(default=None, ge=1, le=64)
    # Local mode only: MATRICXON_MEMORY_SAFETY_MARGIN, default 1.2 (Matricxon's Settings.memory_safety_margin);
    # None = its default. Limited to 1.1-1.8 like Matricxon, validated here so Save rejects bad values instead of
    # failing at the next restart.
    memory_safety_margin: float | None = Field(default=1.2, ge=1.1, le=1.8)
    # Local mode only: MATRICXON_ENABLE_QUANTIZED_NATIVE_COMPUTE (default True, as Matricxon's Settings):
    # computes on quantized weights without a full dequantized copy; fast with the native C kernels, much slower with
    # Numba.
    enable_quantized_native_compute: bool = True
    # Local mode only: MATRICXON_GEMV_BACKEND: "native" (default; C integer kernels compiled with gcc on first
    # start, falls back to numba) or "numba". Needs enable_quantized_native_compute.
    gemv_backend: Literal["numba", "native"] = "native"
    # Local mode only: MATRICXON_TORCH_THREADS, CPU threads for PyTorch and the native kernels. None = every core.
    # Fewer threads means less heat.
    torch_threads: int | None = Field(default=None, ge=1, le=256)
    # Local mode only: MATRICXON_DEVICE, "cpu" (default) or "cuda"/"cuda:N". GPU mode is EXPERIMENTAL (never run
    # on GPU hardware, see../matricxon/ROADMAP.md); a missing GPU fails startup instead of falling back to CPU.
    device: str = Field(default="cpu", pattern=r"^(cpu|cuda(:\d+)?)$")
    # Local mode only, cuda only: MATRICXON_GPU_WEIGHT_MODE, "dequantized" (default; bf16/fp16 in VRAM, fastest,
    # ~2 bytes/param) or "packed" (quantized in VRAM, dequantized per call; less VRAM, slower).
    gpu_weight_mode: Literal["dequantized", "packed"] = "dequantized"
    # Local mode only: MATRICXON_LOG_LEVEL (as Matricxon's Settings.log_level): 0 warnings/errors, 1 per-request
    # milestones, 2 adds a per-layer trace.
    log_level: int = Field(default=0, ge=0, le=2)
    # Local mode only: where saved chat caches live (MATRICXON_PROMPT_CACHE_DIR; key and policy files beside it).
    # None = default inside the project. In a cluster it is the shared folder (Cluster page); existing files are not
    # moved.
    prompt_cache_path: str | None = None
    # Joiner nodes added on the Cluster page; the host pool uses them with the local or remote host(s).
    cluster_hosts: list[str] = Field(default_factory=list, max_length=MAX_REMOTE_HOSTS)
    # How other nodes reach THIS Matricxon (MATRICXON_ADVERTISE_URL, e.g. "http://10.0.0.5:8420"). With cluster
    # mode on, it makes this node the first voter.
    advertise_url: str | None = None
    # The node that starts the cluster, chosen on the Cluster page with the shared folders (one of the hosts); None
    # until chosen.
    first_node: str | None = None
    # Local mode only: MATRICXON_CLUSTER_ROLE (Cluster page on/off): "off" standalone; "seed" the existing install
    # a cluster grows from. Nodes are fresh installs only, never merged (../matricxon/docs/CLUSTER_STORAGE.md).
    # Applies at next start.
    cluster_role: Literal["off", "seed", "joiner"] = "off"
    # Signs pAIring's admin calls to Matricxon (MATRICXON_ADMIN_SECRET on each node). None = nodes are
    # unprotected. Encrypted at rest (settings_service); never sent to the browser.
    admin_secret: str | None = None
    # Nodes that only vote (Matricxon node_kind "voter"): listed with the other nodes but never given chats.
    voter_hosts: list[str] = Field(default_factory=list, max_length=MAX_REMOTE_HOSTS)
    # Overrides for matricxon_installer.install_stream; kept for parity with Ollama/ComfyUI but not functional
    # yet (Matricxon isn't published on GitHub).
    install_repo: str | None = None
    install_version: str | None = None

    @model_validator(mode="after")
    def _fall_back_to_local_when_remote_has_no_hosts(self) -> "MatricxonServerConfig":
        """See OllamaServerConfig's validator."""
        if self.mode == "remote" and not self.remote_hosts:
            self.mode = "local"
        return self


class MatricxonAutoDetectedPath(BaseModel):
    """What matricxon_process._find_project_dir resolves to with no override; the project_dir placeholder
    (like OllamaAutoDetectedPath).
    """

    path: str | None = None
    # What matricxon_process.auto_detect_models_dir resolves to; the models_path placeholder.
    models_path: str | None = None


class MatricxonServerStatus(BaseModel):
    """GET.../status, same shape as OllamaServerStatus. `pid` is read from Matricxon's run/matricxon.pid (written by
    scripts/start.sh)."""

    running: bool
    installed: bool = False
    pid: int | None = None
    healthy: bool | None = None
