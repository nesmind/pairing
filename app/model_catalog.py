"""Curated chat models offered in Settings, beyond what the user pulled manually. Each entry is
*potential*: app/routers/settings.py checks it against what is installed and the hardware
(app/hardware.py) to show "select", "pull" or a disabled row with requirements.

Data lives in `default_models.json` at the repo root (hand-editable, git-tracked), unlike the
admin-managed extended catalog (app.services.extended_model_catalog_service). `CuratedCatalogLoader.load`
validates it; a malformed edit fails loudly at import with a per-field error.

Sizes come from public spec sheets; see ROADMAP.md for upkeep. `min_ram_gb` is generous (~1.25x
weights) for KV cache and runtime overhead.

Each entry has `architecture` (GGUF `general.architecture`) and `quantizations` (the real GGML
tensor types: a mix label like "Q4_K_M" is stored as `["Q4_K", "Q6_K"]`). Matricxon support is
computed live from these against Matricxon's `GET /api/health`
(app.services.matricxon_support_checker.MatricxonSupportChecker), not a copied list.
`vision=True` works on Matricxon when the model's mmproj is pulled there too (narrow support:
one architecture, single image).

Ollama can run every entry regardless.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

_DEFAULT_MODELS_PATH = Path(__file__).resolve().parent.parent / "default_models.json"


@dataclass(frozen=True)
class CuratedModel:
    """One curated, potentially-pullable model; same fields as default_models.json (minus the editor-only `note`)."""

    family: str
    vendor: str
    tag: str
    parameter_size: str
    context_length: int | None
    download_gb: float | None
    min_ram_gb: float
    locally_runnable: bool
    architecture: str
    quantizations: list[str] | None
    vision: bool = False
    unavailable_reason: str | None = None
    # Only on embedding_models entries.
    embedding_dim: int | None = None
    # A vision-projector (mmproj) sidecar, not a chat model (see InstalledProjectorCatalog).
    is_projector: bool = False


class CuratedCatalog:
    """The hand-curated model list: disjoint collections, so embedding entries never reach the chat picker
    (app.services.model_catalog_service) or vice versa.
    """

    def __init__(
        self,
        chat_models: list[CuratedModel],
        embedding_models: list[CuratedModel],
        diffusion_models: list[CuratedModel] | None = None,
        rerank_models: list[CuratedModel] | None = None,
    ) -> None:
        self._chat_models = chat_models
        self._embedding_models = embedding_models
        self._diffusion_models = diffusion_models or []
        self._rerank_models = rerank_models or []

    @property
    def chat_models(self) -> list[CuratedModel]:
        return self._chat_models

    @property
    def embedding_models(self) -> list[CuratedModel]:
        return self._embedding_models

    @property
    def diffusion_models(self) -> list[CuratedModel]:
        """Image (diffusion) models: installed into the stable-diffusion.cpp folder
        (app.services.image_model_service)."""
        return self._diffusion_models

    @property
    def rerank_models(self) -> list[CuratedModel]:
        """Rerankers for Knowledge base retrieval (app.services.document_rerank)."""
        return self._rerank_models

    def find(self, tag: str) -> CuratedModel | None:
        """Chat-model entry by exact tag."""
        return next((entry for entry in self._chat_models if entry.tag == tag), None)

    def find_embedding(self, tag: str) -> CuratedModel | None:
        """Embedding entry by exact tag (see find)."""
        return next((entry for entry in self._embedding_models if entry.tag == tag), None)

    def find_diffusion(self, tag: str) -> CuratedModel | None:
        return next((entry for entry in self._diffusion_models if entry.tag == tag), None)

    def find_rerank(self, tag: str) -> CuratedModel | None:
        return next((entry for entry in self._rerank_models if entry.tag == tag), None)


class _CuratedModelJSON(BaseModel):
    """Validates one default_models.json entry (a hand-edited external boundary). `note` is accepted then dropped."""

    family: str
    vendor: str
    tag: str
    parameter_size: str
    context_length: int | None = None
    download_gb: float | None = None
    min_ram_gb: float
    locally_runnable: bool
    architecture: str
    quantizations: list[str] | None = None
    vision: bool = False
    unavailable_reason: str | None = None
    embedding_dim: int | None = None
    is_projector: bool = False
    note: str | None = None


class CuratedCatalogLoader:
    @staticmethod
    def load(path: Path) -> CuratedCatalog:
        """Reads and validates `path` ({"chat_models", "embedding_models", "diffusion_models", "rerank_models"});
        malformed input raises so a bad edit fails loudly at import instead of emptying the catalog.
        """
        raw = json.loads(path.read_text())
        return CuratedCatalog(
            chat_models=[CuratedCatalogLoader._to_model(entry) for entry in raw["chat_models"]],
            embedding_models=[CuratedCatalogLoader._to_model(entry) for entry in raw["embedding_models"]],
            diffusion_models=[CuratedCatalogLoader._to_model(entry) for entry in raw.get("diffusion_models", [])],
            rerank_models=[CuratedCatalogLoader._to_model(entry) for entry in raw.get("rerank_models", [])],
        )

    @staticmethod
    def _to_model(entry: dict) -> CuratedModel:
        validated = _CuratedModelJSON(**entry)
        return CuratedModel(**validated.model_dump(exclude={"note"}))


CATALOG = CuratedCatalogLoader.load(_DEFAULT_MODELS_PATH)
