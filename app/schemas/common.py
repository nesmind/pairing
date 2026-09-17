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
