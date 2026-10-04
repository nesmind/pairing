"""Shapes shared across more than one domain — defined here (rather than
down in whichever schema file is conceptually closest) since both
schemas/conversation.py and schemas/note.py need PinType, and this file
has no dependency on either."""

from typing import Literal

from pydantic import BaseModel, Field

PinType = Literal["persona", "rules", "skill"]

# Shared by app.schemas.ollama_server_config.OllamaServerConfig and
# app.schemas.comfyui_config.ComfyUIProcessConfig — "local" means this
# app itself owns the process (spawns/kills it, see
# app.services.ollama_process.py/comfyui_process.py); "remote" means the
# admin points at 1-12 external hosts instead and the app load-balances
# across them (see app.services.server_pool.HostPool).
ServerMode = Literal["local", "remote"]

# Which LLM backend app.services.inference_client dispatches chat/embedding/model-management calls to right now
# — see app.services.engine_service, which owns *live* engine state (the DB-backed setting/its in-process
# cache); this is just the type + default it's typed against. Only one is ever active at a time; the others can
# still be configured/started/stopped (Ollama/Matricxon) or configured via a Connector (RunPod — see
# app.services.connectors) independently, they just don't serve live traffic. Kept in sync with
# app.services.engines.registry by tests/test_engine_registry.py.
EngineName = Literal["ollama", "matricxon", "runpod"]

# The single source of truth for "no engine explicitly chosen yet" — used both as ActiveEngineConfig's own
# default below and as app.services.engine_service.DEFAULT_ENGINE (imported from here, not redefined there).
# Matricxon while it's still the engine under active development/testing — revisit once Ollama should go back
# to being the safer default. Previously drifted into two separate, disagreeing copies (this field defaulted
# to "ollama" while engine_service.DEFAULT_ENGINE said "matricxon") — a fresh install with no saved setting at
# all silently routed to whichever one a given code path happened to ask, confirmed live, 2026-09-27.
DEFAULT_ENGINE: EngineName = "matricxon"


class ActiveEngineConfig(BaseModel):
    """GET/PUT /api/settings/engine body — see app.services.engine_service.get_active_engine/set_active_engine."""

    active_engine: EngineName = DEFAULT_ENGINE


class EngineCapabilitiesOut(BaseModel):
    """Pydantic mirror of app.services.engines.base.EngineCapabilities (a plain dataclass — services code has
    no reason to depend on Pydantic) — see that class's own docstring for what each flag means. Built via
    `EngineCapabilitiesOut(**dataclasses.asdict(engine.capabilities))` at the one call site that needs this
    (GET /api/settings/engine/options)."""

    embeddings: bool
    model_management: bool
    stop_model: bool
    local_process: bool
    host_pool: bool
    support_checking: bool
    format_introspection: bool


class EngineOption(BaseModel):
    """One entry in GET /api/settings/engine/options — every engine app.services.engines.registry knows about,
    for a caller that wants to render a picker/dropdown or check a capability without hardcoding engine names
    (see Settings > External servers' "Active engine" picker and the Stats page's engine dropdown, both
    previously hand-written <option> lists)."""

    name: EngineName
    display_name: str
    capabilities: EngineCapabilitiesOut
    # Whether this engine can actually be activated right now (see
    # app.services.engines.base.InferenceEngine.is_ready) — False for a Connector-backed engine (e.g. "runpod")
    # an admin hasn't configured/enabled yet on the Connectors page. Always True for Ollama/Matricxon.
    ready: bool = True


class EngineOptionsResponse(BaseModel):
    engines: list[EngineOption]


# Shared by the same two schemas above (their remote_hosts field's own
# max_length) and app.services.settings_service.get_ollama_server_config's
# legacy .env fallback truncation — one place to change the cap for both
# services rather than two numbers that could silently drift apart.
MAX_REMOTE_HOSTS = 120


class InstallDefaults(BaseModel):
    """The maintainer-picked GitHub repo/version app.services.ollama_installer
    or comfyui_installer falls back to when an admin hasn't overridden
    OllamaServerConfig.install_repo/install_version (or ComfyUIProcessConfig's
    identical pair) — read-only, exposed so Settings > External servers
    can show "what will actually be installed" next to the Install
    button and pre-fill the override fields' placeholders."""

    repo: str
    version: str


class AvailableVersions(BaseModel):
    """GET .../available-versions response — real tags fetched live from GitHub (see
    app.services.github_releases.list_tags), newest first, for Settings > External servers' version picker to
    offer as a datalist alongside the free-text override field (a repo an admin points at might have no tags at
    all, or one they want by branch/SHA instead — so this is always a convenience on top of manual entry, never
    a replacement for it). Empty when GitHub couldn't be reached or the repo has no tags — the frontend falls
    back to plain manual entry in that case, same as before this endpoint existed."""

    versions: list[str]


class HostHealthCheck(BaseModel):
    """GET .../check-host?host=... response — a direct, one-off probe of
    `host` (see app.services.server_pool.ping_host), not limited to hosts
    already saved to the live pool. Settings > External servers uses one
    of these per remote-host row, so an admin can verify a candidate URL
    is actually reachable before saving the form."""

    host: str
    healthy: bool


class OkResponse(BaseModel):
    """Generic acknowledgement body for endpoints whose only job is to
    say "that worked" — deleting something, changing a password, and the
    like. A named schema (per CLAUDE.md: never return raw dicts) rather
    than a per-endpoint one-field model, since every one of these really
    is just this same one boolean."""

    ok: bool = True


class GenerationParams(BaseModel):
    """The full set of tunable knobs for one conversation. Mirrors
    app.config.DEFAULT_GENERATION_PARAMS — see that file for what each
    field actually does to the model's output."""

    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(40, ge=1, le=200)
    repeat_penalty: float = Field(1.1, ge=0.0, le=2.0)
    num_ctx: int = Field(4096, ge=256, le=32768)
    num_predict: int = Field(1024, ge=-1, le=8192)
    seed: int = -1
    # Independent of which chat `model` the conversation uses — every
    # model shares the same knowledge base and retrieval settings (see
    # app/services/document_service.py), so switching models never
    # changes what RAG behavior does. Only the model's own generation
    # knobs above vary per model.
    rag_top_k: int = Field(4, ge=1, le=10)
    # Which of this conversation's persona/rules/skill slots have been
    # explicitly turned off from the chat page (see
    # GET/DELETE /api/notes/slots/{conversation_id}/...) — a turned-off
    # slot shows no icon and contributes nothing to the prompt, instead
    # of falling back to the user's default note for that slot (see
    # app.services.note_service.resolve_conversation_notes). Declared as
    # a real field (not just an ad-hoc dict key) specifically so a
    # normal Settings-page save — which replaces `params` wholesale from
    # this model's own model_dump() — doesn't silently wipe it out.
    disabled_default_notes: list[PinType] = []
    # Offer the admin-connected MCP tools to the model in this conversation (see tool_loop_service).
    use_tools: bool = False
    # The tool definitions this chat offers its model, saved the first time it uses tools (see
    # app.services.chat_tool_set). Server-managed: a params update from a client never changes it. Declared so a
    # Settings-page save, which rebuilds `params` from this model, doesn't lose it.
    tool_snapshot: list[dict] = []
