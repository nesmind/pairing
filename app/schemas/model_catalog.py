"""Request/response shapes for Settings > Model — the default catalog (app/model_catalog.py), the admin-managed
"browse more models" extended catalog (app/services/extended_model_catalog_service.py), and Hugging Face
search/lookup (app/services/huggingface_client.py). Split out of app/schemas/settings.py once this many
model-catalog-specific shapes pushed that file over CLAUDE.md's line cap."""

from pydantic import BaseModel

from app.schemas.settings import HardwareSummary


class CatalogEntry(BaseModel):
    """One model Settings can offer — either already pulled into Ollama,
    pullable-and-compatible, or blocked by hardware. See
    app/model_catalog.py for where these come from and
    app/services/settings_service.py for how `installed`/`hardware_ok`
    are computed."""

    family: str
    vendor: str
    tag: str | None
    parameter_size: str
    context_length: int | None = None
    download_gb: float | None = None
    min_ram_gb: float
    locally_runnable: bool
    installed: bool
    hardware_ok: bool
    unavailable_reason: str | None = None
    # Whether this model can see an image attachment (see
    # app.services.chat_attachment_service and the "Default vision
    # model" picker in Settings) — for a curated app/model_catalog.py
    # entry, hand-annotated there; for an already-installed model not in
    # that list, read live from Ollama's own "vision" capability tag.
    vision: bool = False
    # Whether this model can also hold an ordinary text conversation —
    # true for every entry this catalog can currently produce (see
    # build_model_catalog's own "completion" capability filter, which
    # every entry here already passed), so `vision and text_capable`
    # ("+Vision", a normal chat model that also sees images, e.g.
    # gemma4:12b) is the only combination reachable today; `vision and
    # not text_capable` ("Vision" alone, a vision-only model with no
    # text chat ability) is modeled honestly for correctness but can't
    # actually appear in this catalog as currently scoped.
    text_capable: bool = True
    # Whether an admin has hidden this model from regular users' picker
    # (see POST /api/settings/hide-model). Non-admin callers never
    # receive a hidden entry at all — this field only ever comes back
    # `true` for an admin, so their own catalog view can show which
    # models they've hidden and offer an "Unhide" action.
    hidden: bool = False
    # Whether "Remove from list" should be offered (admin-only, see DELETE /api/settings/model-catalog/extended)
    # — true only for a not-yet-installed entry from the admin-managed extended catalog (see
    # app.services.extended_model_catalog_service.build_extended_catalog); always False for a hand-curated
    # app/model_catalog.py default (there's nothing stored to delete) and for an installed model (that gets
    # "Uninstall" instead — removing it from a *list* while it's still actually pulled into Ollama would just be
    # confusing, and re-adding it later would need a redundant Hugging Face re-verification for no reason).
    removable: bool = False


class ModelCatalogResponse(BaseModel):
    entries: list[CatalogEntry]
    hardware: HardwareSummary


class EmbeddingCatalogEntry(BaseModel):
    """One embedding model from app/model_catalog.py's EMBEDDING_CATALOG — the embedding-model analogue of
    CatalogEntry above, kept as its own type rather than reusing that one since several of its fields (hidden,
    removable, text_capable, vision) are meaningless for an embedding-only model."""

    family: str
    vendor: str
    tag: str
    parameter_size: str
    context_length: int | None = None
    # Output vector size (768 for nomic-embed-text, 384 for MiniLM) — not meaningful for a chat model, so kept
    # here rather than added to CatalogEntry.
    embedding_dim: int
    download_gb: float | None = None
    min_ram_gb: float
    installed: bool
    hardware_ok: bool
    unavailable_reason: str | None = None


class EmbeddingModelCatalogResponse(BaseModel):
    entries: list[EmbeddingCatalogEntry]
    # Whichever tag app.config.EMBEDDING_MODEL currently resolves to, so the UI can label which entry is
    # actually in use without every entry needing its own is_default flag.
    default_tag: str
    hardware: HardwareSummary


class InstalledModelsResponse(BaseModel):
    """GET /api/settings/installed-models — the chat page's model-switch
    picker on an existing conversation, which only needs plain tags, not
    get_model_catalog's hardware/download info."""

    models: list[str]


class PullModelRequest(BaseModel):
    tag: str


class HideModelRequest(BaseModel):
    tag: str
    hidden: bool


class HideModelResponse(BaseModel):
    tag: str
    hidden: bool


class DeleteModelResponse(BaseModel):
    deleted: str


class ExtendedModelCatalogResponse(BaseModel):
    """GET /api/settings/model-catalog/extended — the admin-managed "browse more models" list behind Settings >
    Model (see app.services.extended_model_catalog_service), separate from the small hand-curated default list
    GET /api/settings/model-catalog still returns unchanged."""

    entries: list[CatalogEntry]


class AddExtendedModelRequest(BaseModel):
    """POST /api/settings/model-catalog/extended — adds one specific GGUF file from a Hugging Face repo (see
    app.services.extended_model_catalog_service.add_model, which re-verifies both against Hugging Face itself
    rather than trusting this request body)."""

    repo_id: str
    filename: str


class RemoveExtendedModelRequest(BaseModel):
    tag: str


class SearchHfModelsRequest(BaseModel):
    query: str


class HfSearchResult(BaseModel):
    """One repo from POST /api/settings/model-catalog/extended/search-hf — see
    app.services.huggingface_client.search_models. `license` is always None here — Hugging Face's own search API
    doesn't return it; only the per-repo lookup (HfRepoFilesResponse below) does."""

    repo_id: str
    downloads: int
    likes: int
    gated: bool
    license: str | None


class SearchHfModelsResponse(BaseModel):
    results: list[HfSearchResult]


class HfFileOption(BaseModel):
    filename: str
    download_gb: float


class HfRepoFilesRequest(BaseModel):
    repo_id: str


class HfRepoFilesResponse(BaseModel):
    """POST /api/settings/model-catalog/extended/hf-files — every single-file GGUF variant `repo_id` offers,
    each with its real download size, plus repo-level metadata (see
    app.services.huggingface_client.get_repo_files). The Model tab's own file-picker step, shown once an admin
    picks a repo from a search result (or already knows the exact repo path)."""

    repo_id: str
    family: str | None
    parameter_size: str | None
    context_length: int | None
    gated: bool
    license: str | None
    files: list[HfFileOption]
