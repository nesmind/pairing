"""Shapes for the image-model store (see app/services/image_model_service.py, app/routers/image_model_admin.py)."""

from pydantic import BaseModel

from app.schemas.model_catalog import CatalogEntry


class ImageModelFile(BaseModel):
    """One diffusion model file in the stable-diffusion.cpp models folder. `path` is what the engine form saves
    as its Model; `tag` is set only for a file downloaded through Settings > Model (its hf.co/... catalog tag)."""

    name: str
    path: str
    size_gb: float
    tag: str | None = None


class ImageModelList(BaseModel):
    models: list[ImageModelFile]
    # False while the engine is in Remote mode: models are managed on the remote host, not downloaded here.
    local: bool = True
    # Where models are stored (see app.services.image_model_service.resolve_models_dir) — the form's placeholder.
    folder: str = ""


class DiffusionModelCatalogResponse(BaseModel):
    """Settings > Model's "Diffusion models" section: the curated defaults (default_models.json), admin-added
    Hugging Face image models, and anything else installed — each with its install state. Same for every chat engine."""

    entries: list[CatalogEntry]
    # False while the image engine is in Remote mode: models are managed on the remote host, so nothing is offered.
    local: bool = True


class ImageModelRequest(BaseModel):
    tag: str


class CompanionPullRequest(BaseModel):
    """The companion files (by flag, e.g. "--llm") the user chose to download for a model."""

    tag: str
    flags: list[str]
