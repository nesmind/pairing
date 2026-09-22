"""
Every embedding model in app/model_catalog.py's CATALOG.embedding_models, cross-referenced with what's actually
pulled into Ollama — the embedding-model analogue of app.services.model_catalog_service.ChatModelCatalogBuilder,
kept in its own file (both for CLAUDE.md's file-size rule and because embedding_models is deliberately a
disjoint list — see CuratedCatalog's own docstring) since it has no hidden-tags/per-user logic worth sharing.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app import hardware
from app.config import EMBEDDING_MODEL
from app.model_catalog import CATALOG
from app.schemas import EmbeddingCatalogEntry, EmbeddingModelCatalogResponse
from app.services.default_model_settings import DefaultModelSettings
from app.services.inference_client import list_models
from app.services.matricxon_support_checker import MatricxonSupportChecker
from app.services.model_catalog_service import OLLAMA_RAM_ESTIMATE_MULTIPLIER


class EmbeddingModelCatalogService:
    """Open to any logged-in user, same visibility as GET /api/settings/rag-availability — installed status
    isn't admin-secret, only pulling is."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def build(self) -> EmbeddingModelCatalogResponse:
        installed_models = await list_models()
        installed_by_tag = {m["name"]: m for m in installed_models if "embedding" in m.get("capabilities", [])}
        capacity_gb = hardware.available_capacity_gb()
        checker = await MatricxonSupportChecker.load(installed_models)

        entries = []
        for entry in CATALOG.embedding_models:
            installed = entry.tag in installed_by_tag
            verdict = checker.verdict_for(entry.architecture, entry.quantizations)
            min_ram_gb_ollama = (
                round(entry.download_gb * OLLAMA_RAM_ESTIMATE_MULTIPLIER, 1) if entry.download_gb else entry.min_ram_gb
            )
            entries.append(
                EmbeddingCatalogEntry(
                    family=entry.family,
                    vendor=entry.vendor,
                    tag=entry.tag,
                    parameter_size=entry.parameter_size,
                    context_length=entry.context_length,
                    embedding_dim=entry.embedding_dim,
                    download_gb=entry.download_gb,
                    min_ram_gb=entry.min_ram_gb,
                    min_ram_gb_ollama=min_ram_gb_ollama,
                    min_ram_gb_matricxon=checker.estimated_ram_gb(entry.tag),
                    installed=installed,
                    hardware_ok=installed or capacity_gb >= entry.min_ram_gb,  # same exemption as above
                    unavailable_reason=entry.unavailable_reason,
                    matricxon_supported=verdict.supported,
                    matricxon_unsupported_reason=verdict.reason,
                )
            )

        # The admin's actual configured choice when there is one installed to point at; falls back to the
        # app's own built-in default tag otherwise, purely so the "Default" badge still guides a fresh install
        # toward *something* worth pulling before anything's been configured or installed yet.
        default_tag = (await DefaultModelSettings(self._db).embedding()) or EMBEDDING_MODEL

        return EmbeddingModelCatalogResponse(
            entries=entries, default_tag=default_tag, hardware=hardware.hardware_summary()
        )
