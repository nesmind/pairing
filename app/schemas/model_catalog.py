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
    # The hardware-gating threshold a not-yet-installed entry is checked against (see
    # ChatModelCatalogBuilder's hardware_ok / app.routers.settings's pull-time capacity check) — this is
    # app/model_catalog.py's own hand-typed figure, kept as-is for that purpose. For *display*, prefer
    # min_ram_gb_ollama/min_ram_gb_matricxon below — real, computed-per-engine numbers, not a
    # hand-typed guess that can drift from either engine's actual real requirement (confirmed live,
    # 2026-09-21: Matricxon dequantizes to bf16 before computing, so its real need for a real
    # Ministral-3B Q4_K_M file runs to ~7.5GB, not the ~3GB this field alone implied).
    min_ram_gb: float
    # Ollama's own real per-model RAM figure — computed from this entry's real download_gb (Ollama's
    # API exposes no better number of its own), or falling back to min_ram_gb only when download_gb
    # isn't known at all. See app.services.model_catalog_service.OLLAMA_RAM_ESTIMATE_MULTIPLIER.
    min_ram_gb_ollama: float | None = None
    # Matricxon's own real, current per-tag `estimated_ram_gb` (see
    # ../matricxon/app/models/load_dtype.py's own docstring) — only known once this exact tag is
    # confirmed actually installed on Matricxon (its real GGUF tensor shapes have to be read to
    # compute it); None otherwise, including whenever Matricxon can't be reached at all. Never a
    # guess extrapolated from Ollama's own figure — the two engines' real per-model requirements are
    # not proportional to each other (dequantize-to-bf16-then-compute vs. quantized-native compute).
    min_ram_gb_matricxon: float | None = None
    locally_runnable: bool
    installed: bool
    hardware_ok: bool
    unavailable_reason: str | None = None
    # Whether this model can see an image attachment (see
    # app.services.chat_attachment_service and the "Default vision
    # model" picker in Settings). For a not-yet-installed curated
    # app/model_catalog.py entry, hand-annotated there as a pre-install
    # estimate only; once a tag is actually installed (curated or not),
    # read live from the active engine's own "vision" capability tag
    # instead — confirmed live, 2026-09-22: the curated flag reflects
    # Ollama's own auto-pairing behavior for a hf.co/ pull and can
    # disagree with what's actually installed under a different engine
    # (e.g. Matricxon, which doesn't auto-pair a projector the same way).
    vision: bool = False
    # Whether this model can also hold an ordinary text conversation —
    # true for every entry this catalog can currently produce (see
    # ChatModelCatalogBuilder's own "completion" capability filter, which
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
    # app.services.extended_model_catalog_service.ExtendedModelCatalog.build); always False for a hand-curated
    # app/model_catalog.py default (there's nothing stored to delete) and for an installed model (that gets
    # "Uninstall" instead — removing it from a *list* while it's still actually pulled into Ollama would just be
    # confusing, and re-adding it later would need a redundant Hugging Face re-verification for no reason).
    removable: bool = False
    # Whether this model's architecture+quantization is actually implemented by the Matricxon engine (see
    # ../matricxon/app/architectures/registry.py and .../gguf/dequant/registry.py) — independent of Ollama, which
    # can run every entry this catalog offers. For a hand-curated app/model_catalog.py entry, hand-verified True
    # or False against those two registries (see that entry's own comment for which and why). Defaults True here
    # only as the field's bare fallback; an admin-added extended-catalog entry (arbitrary Hugging Face repo, no
    # verified architecture) is explicitly constructed with this False instead — see
    # app.services.extended_model_catalog_service.ExtendedModelCatalog.build's own comment — since Matricxon fails
    # closed on an unrecognized architecture and a false "supported" badge would be actively misleading.
    matricxon_supported: bool = True
    matricxon_unsupported_reason: str | None = None
    # A vision-projector (mmproj) sidecar, pulled alongside its own paired text model rather than run as a
    # standalone chat model — see
    # app.services.huggingface_client.HuggingFaceCatalogSearch.is_projector_file's own docstring. Lets the
    # frontend skip the matricxon_supported "Not supported" badge for one of these specifically: that badge
    # means "Matricxon can't run this as a chat model," which is a true but actively misleading thing to say
    # about a file that was never meant to be run as one on its own — confirmed live, it read as "Matricxon
    # can't handle this," when the paired text model this projector exists to be paired with may well be
    # supported on its own (see MatricxonSupportChecker.verdict_for's own docstring on is_projector).
    is_projector: bool = False
    # True for an installed model that isn't in app/model_catalog.py's hand-curated list at all — surfaced
    # through ChatModelCatalogBuilder._auto_discovered_entry, InstalledProjectorCatalog.entries, or
    # ExtendedModelCatalog.build (the latter always paired with installed=False, so it never actually changes
    # what the frontend shows there). Lets the frontend label a stray/manually-added install ("Not in catalog")
    # instead of it looking like a hand-curated entry with broken/missing data — confirmed live (2026-09-21)
    # that a duplicate model pulled under an unfamiliar repo name showed up as a bare "Phi2 · unknown" row with
    # nothing explaining why it looked so different from every other entry.
    is_auto_discovered: bool = False


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
    # See CatalogEntry.min_ram_gb_ollama/min_ram_gb_matricxon's own docstrings — same meaning here.
    min_ram_gb_ollama: float | None = None
    min_ram_gb_matricxon: float | None = None
    installed: bool
    hardware_ok: bool
    unavailable_reason: str | None = None
    # See CatalogEntry.matricxon_supported/matricxon_unsupported_reason's own docstring — same meaning here.
    matricxon_supported: bool = True
    matricxon_unsupported_reason: str | None = None


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
    # True once the admin has confirmed "pull anyway" past a 409 DuplicateInstallDetector warning (see
    # app/routers/settings.py's pull_model) — False on a normal first attempt, so the duplicate check always
    # runs at least once.
    confirm_duplicate: bool = False


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
    # POST only (see add_extended_model's own docstring) — True when the just-added tag turned out to already
    # be installed, so it was never going to appear in `entries` above (ExtendedModelCatalog.build excludes any
    # installed tag on purpose — see its own docstring). Lets the frontend tell that apart from a genuine new
    # addition instead of showing a flat "Added" that's misleading here: confirmed live, an admin re-adding an
    # already-installed tag saw "Added ..." and then couldn't find it anywhere to pull, since there was nothing
    # left to pull. Always False on GET/DELETE, where it's meaningless.
    already_installed: bool = False


class AddExtendedModelRequest(BaseModel):
    """POST /api/settings/model-catalog/extended — adds one specific GGUF file from a Hugging Face repo (see
    app.services.extended_model_catalog_service.ExtendedModelCatalog.add, which re-verifies both against
    Hugging Face itself rather than trusting this request body)."""

    repo_id: str
    filename: str


class RemoveExtendedModelRequest(BaseModel):
    tag: str


class SearchHfModelsRequest(BaseModel):
    query: str


class HfSearchResult(BaseModel):
    """One repo from POST /api/settings/model-catalog/extended/search-hf — see
    app.services.huggingface_client.HuggingFaceCatalogSearch.search. `license` is always None here — Hugging
    Face's own search API doesn't return it; only the per-repo lookup (HfRepoFilesResponse below) does."""

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
    # See app.services.huggingface_client.HuggingFaceCatalogSearch.is_projector_file's own docstring — a
    # vision-projector (mmproj) sidecar, not a standalone chat model, so the file-picker UI flags it rather than
    # implying it's whatever this response's own family/parameter_size (below) describes, which almost never
    # applies to it.
    is_projector: bool = False
    # A chat-model file whose repo also ships an mmproj sidecar — see HuggingFaceCatalogSearch.repo_files.
    vision: bool = False


class HfRepoFilesRequest(BaseModel):
    repo_id: str


class HfRepoFilesResponse(BaseModel):
    """POST /api/settings/model-catalog/extended/hf-files — every single-file GGUF variant `repo_id` offers,
    each with its real download size, plus repo-level metadata (see
    app.services.huggingface_client.HuggingFaceCatalogSearch.repo_files). The Model tab's own file-picker step,
    shown once an admin picks a repo from a search result (or already knows the exact repo path)."""

    repo_id: str
    family: str | None
    parameter_size: str | None
    context_length: int | None
    gated: bool
    license: str | None
    files: list[HfFileOption]
