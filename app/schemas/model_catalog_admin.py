"""Admin request/response shapes for Settings > Model: pull/hide/delete, the extended "browse more models"
catalog (extended_model_catalog_service.py) and Hugging Face search (huggingface_client.py). Split from
schemas/model_catalog.py for the file-size cap; all re-exported from app.schemas.
"""

from pydantic import BaseModel

from app.schemas.model_catalog import CatalogEntry


class PullModelRequest(BaseModel):
    tag: str
    # True once the admin confirmed "pull anyway" after a 409 DuplicateInstallDetector warning
    # (routers/settings.py pull_model); False on a first attempt, so the check always runs.
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
    """GET /api/settings/model-catalog/extended: the admin-managed "browse more models" list
    (extended_model_catalog_service), separate from the curated GET /api/settings/model-catalog.
    """

    entries: list[CatalogEntry]
    # POST only: true when the added tag is already installed, so it never appears in `entries`
    # (ExtendedModelCatalog.build excludes installed tags). Lets the frontend avoid a misleading "Added"
    # (an admin re-adding an installed tag found nothing to pull). False on GET/DELETE.
    already_installed: bool = False


class AddExtendedModelRequest(BaseModel):
    """POST /api/settings/model-catalog/extended: adds one GGUF file from a Hugging Face repo;
    ExtendedModelCatalog.add re-verifies both against Hugging Face.
    """

    repo_id: str
    filename: str


class RemoveExtendedModelRequest(BaseModel):
    tag: str


class SearchHfModelsRequest(BaseModel):
    query: str


class HfSearchResult(BaseModel):
    """One repo from POST.../extended/search-hf (HuggingFaceCatalogSearch.search). `license` is always None:
    the search API omits it; only the per-repo lookup (HfRepoFilesResponse) has it.
    """

    repo_id: str
    downloads: int
    likes: int
    gated: bool
    license: str | None
    # A text-to-image repo: installs into the image engine.
    is_image: bool = False


class SearchHfModelsResponse(BaseModel):
    results: list[HfSearchResult]


class HfFileOption(BaseModel):
    filename: str
    download_gb: float
    # A vision-projector (mmproj) sidecar, not a chat model (HuggingFaceCatalogSearch.is_projector_file); the
    # file picker flags it, since family/parameter_size rarely apply to it.
    is_projector: bool = False
    # A chat-model file whose repo also ships an mmproj sidecar (HuggingFaceCatalogSearch.repo_files).
    vision: bool = False


class HfRepoFilesRequest(BaseModel):
    repo_id: str


class HfRepoFilesResponse(BaseModel):
    """POST.../extended/hf-files: every single-file GGUF variant of `repo_id` with real sizes, plus repo
    metadata (HuggingFaceCatalogSearch.repo_files); the Model tab's file-picker step.
    """

    repo_id: str
    family: str | None
    parameter_size: str | None
    context_length: int | None
    gated: bool
    license: str | None
    is_image: bool = False
    files: list[HfFileOption]
