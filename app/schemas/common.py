"""Shapes shared across domains (both schemas/conversation.py and note.py need PinType), with no dependency on
either."""

from typing import Literal

from pydantic import BaseModel, Field

PinType = Literal["persona", "rules", "skill"]

# Shared by OllamaServerConfig, MatricxonServerConfig and SdCppConfig: "local" = this app owns the process
# (ollama_process.py/sdcpp_process.py); "remote" = 1-12 external hosts, load-balanced (server_pool.HostPool).
ServerMode = Literal["local", "remote"]

# Which backend inference_client dispatches chat/embedding/model-management to. Live state is in
# engine_service; this is the type and default. One engine is active; others can still be configured or
# run, or set up via a Connector (RunPod, services.connectors). Kept in sync with engines.registry by
# tests/test_engine_registry.py.
EngineName = Literal[
    "ollama",
    "matricxon",
    "runpod",
    "cloudflare",
    "digitalocean",
    "openrouter",
    "gemini",
    "groq",
    "nvidia",
    "cohere",
]

# Single source for "no engine chosen yet", used by ActiveEngineConfig and engine_service.DEFAULT_ENGINE.
# Matricxon while it is under active development; revisit when Ollama should be the safer default.
# (Two disagreeing copies once routed fresh installs inconsistently,.)
DEFAULT_ENGINE: EngineName = "matricxon"


class ActiveEngineConfig(BaseModel):
    """GET/PUT /api/settings/engine body (engine_service.get_active_engine/set_active_engine)."""

    active_engine: EngineName = DEFAULT_ENGINE


class EngineCapabilitiesOut(BaseModel):
    """Pydantic mirror of engines.base.EngineCapabilities (a dataclass, so services avoid Pydantic), built via
    `EngineCapabilitiesOut(**dataclasses.asdict(engine.capabilities))` for GET /api/settings/engine/options.
    """

    embeddings: bool
    model_management: bool
    stop_model: bool
    local_process: bool
    host_pool: bool
    support_checking: bool
    format_introspection: bool
    tools: bool = False
    vision: bool = False


class EngineOption(BaseModel):
    """One entry of GET /api/settings/engine/options: every registered engine, for pickers and capability
    checks without hardcoded names (Active engine picker, Stats engine dropdown).
    """

    name: EngineName
    display_name: str
    capabilities: EngineCapabilitiesOut
    # Whether the engine can be activated now (InferenceEngine.is_ready); False for an unconfigured Connector
    # engine (e.g. "runpod"). Always True for Ollama/Matricxon.
    ready: bool = True


class EngineOptionsResponse(BaseModel):
    engines: list[EngineOption]


# Cap shared by both schemas' remote_hosts max_length and settings_service.get_ollama_server_config's
# legacy.env truncation.
MAX_REMOTE_HOSTS = 120


class InstallDefaults(BaseModel):
    """The GitHub repo/version an engine installer (ollama/matricxon/sdcpp) uses unless overridden; shown in Settings
    > Inference engines next to Install and as placeholders.
    """

    repo: str
    version: str


class AvailableVersions(BaseModel):
    """GET.../available-versions: tags fetched live from GitHub (github_releases.list_tags), newest first,
    for the version datalist. A convenience on top of manual entry (a repo may have no tags); empty if
    GitHub is unreachable.
    """

    versions: list[str]


class HostHealthCheck(BaseModel):
    """GET.../check-host?host=...: a one-off probe of `host` (server_pool.ping_host), even if not in the
    live pool; one per remote-host row so admins can verify a URL before saving.
    """

    host: str
    healthy: bool


class OkResponse(BaseModel):
    """Generic acknowledgement for endpoints that only confirm success (delete, password change); named
    because CLAUDE.md forbids raw dicts.
    """

    ok: bool = True


class GenerationParams(BaseModel):
    """All tunable knobs for one conversation; mirrors app.config.DEFAULT_GENERATION_PARAMS."""

    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(40, ge=1, le=200)
    repeat_penalty: float = Field(1.1, ge=0.0, le=2.0)
    num_ctx: int = Field(4096, ge=256, le=32768)
    num_predict: int = Field(1024, ge=-1, le=8192)
    seed: int = -1
    # Independent of the chat `model`: all models share the knowledge base and retrieval settings
    # (document_service.py). Only the generation knobs above vary per model.
    rag_top_k: int = Field(4, ge=1, le=10)
    # Persona/rules/skill slots turned off from the chat page (GET/DELETE /api/notes/slots/{conversation_id}/...):
    # no icon, nothing in the prompt (note_service.resolve_conversation_notes). A real field so a Settings save,
    # which rebuilds `params` from this model, does not wipe it.
    disabled_default_notes: list[PinType] = []
    # Offer the admin-connected MCP tools in this conversation (tool_loop_service).
    use_tools: bool = False
    # Tool definitions this chat offers, saved at first use (chat_tool_set). Server-managed: client updates
    # never change it. Declared so a Settings save does not lose it.
    tool_snapshot: list[dict] = []
