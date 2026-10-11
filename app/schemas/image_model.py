"""Shapes for the image-model store (services/image_model_service.py, routers/image_model_admin.py)."""

from pydantic import BaseModel

from app.schemas.model_catalog import CatalogEntry


class ImageModelFile(BaseModel):
    """A diffusion model file in the stable-diffusion.cpp models folder. `path` is what the engine form saves
    as Model; `tag` is set only for files downloaded via Settings > Model (hf.co/... tag).
    """

    name: str
    path: str
    size_gb: float
    tag: str | None = None


class ImageModelList(BaseModel):
    models: list[ImageModelFile]
    # False in Remote mode: models are managed on the remote host.
    local: bool = True
    # Model storage (image_model_service.resolve_models_dir); the form's placeholder.
    folder: str = ""


class DiffusionModelCatalogResponse(BaseModel):
    """Settings > Model's "Diffusion models": curated defaults (default_models.json), admin-added Hugging Face
    models and other installed ones, with install state. Same for every chat engine.
    """

    entries: list[CatalogEntry]
    # False in Remote mode: models are managed on the remote host, so none are offered.
    local: bool = True


class ImageModelRequest(BaseModel):
    tag: str


class CompanionPullRequest(BaseModel):
    """The companion files (by flag, e.g. "--llm") the user chose to download for a model."""

    tag: str
    flags: list[str]
