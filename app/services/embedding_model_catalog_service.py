"""Embedding models from CATALOG.embedding_models cross-referenced with what the engine has pulled; the
analogue of model_catalog_service.ChatModelCatalogBuilder. Own file for the file-size rule and because
embedding_models is a disjoint list (CuratedCatalog) with no hidden-tags/per-user logic to share.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app import hardware
from app.config import EMBEDDING_MODEL
from app.model_catalog import CATALOG
from app.schemas import EmbeddingCatalogEntry, EmbeddingModelCatalogResponse
from app.services.default_model_settings import DefaultModelSettings
from app.services.engine_support_checker import EngineSupportSet
from app.services.inference_client import list_models
from app.services.model_catalog_service import ChatModelCatalogBuilder


class EmbeddingModelCatalogService:
    """Open to any logged-in user, like GET /api/settings/rag-availability: install state isn't secret, only pulling
    is."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def build(self) -> EmbeddingModelCatalogResponse:
        installed_models = await list_models()
        installed_by_tag = {m["name"]: m for m in installed_models if "embedding" in m.get("capabilities", [])}
        capacity_gb = hardware.available_capacity_gb()
        support_set = await EngineSupportSet.load(installed_models)

        in_use = await DefaultModelSettings(self._db).embedding()
        entries = []
        for entry in CATALOG.embedding_models:
            installed_model = installed_by_tag.get(entry.tag)
            installed = installed_model is not None
            # Once installed, the engine's live report is the only source of architecture, with no fallback to the
            # curated guess (same as ChatModelCatalogBuilder._catalog_entry): the static value is a pre-install
            # estimate and must not outrank what is running.
            architecture = (
                ChatModelCatalogBuilder._clean_engine_reported(installed_model.get("details", {}).get("family"))
                if installed_model is not None
                else entry.architecture
            )
            engine_support = support_set.support_for(
                tag=entry.tag,
                architecture=architecture,
                quantizations=entry.quantizations,
                download_gb=entry.download_gb,
                min_ram_gb=entry.min_ram_gb,
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
                    engine_support=engine_support,
                    installed=installed,
                    hardware_ok=installed or capacity_gb >= entry.min_ram_gb,
                    unavailable_reason=entry.unavailable_reason,
                    in_use=entry.tag == in_use,
                )
            )

        # The admin's choice if installed; otherwise the built-in default tag, so the "Default" badge points a
        # fresh install at something to pull.
        default_tag = in_use or EMBEDDING_MODEL

        return EmbeddingModelCatalogResponse(
            entries=entries, default_tag=default_tag, hardware=hardware.hardware_summary()
        )
