"""Admin-facing request/response shapes for Settings > Model: pulling/hiding/deleting a model, the admin-managed
"browse more models" extended catalog (app/services/extended_model_catalog_service.py), and Hugging Face
search/lookup (app/services/huggingface_client.py). Split out of app/schemas/model_catalog.py (the plain
catalog-entry shapes every caller reads, CatalogEntry/EmbeddingCatalogEntry/EngineModelSupport) once this file
pushed that one over CLAUDE.md's line cap — every one of these is re-exported from app.schemas unchanged, so no
import site elsewhere needed to change."""

from pydantic import BaseModel

from app.schemas.model_catalog import CatalogEntry


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
