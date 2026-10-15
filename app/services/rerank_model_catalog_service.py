"""Settings > Models > Reranker models: CATALOG.rerank_models with install state, like the embedding list."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.model_catalog import CATALOG
from app.schemas.rerank import RerankCatalogEntry, RerankModelCatalogResponse
from app.services.default_model_settings import DefaultModelSettings
from app.services.engines.registry import registry


class RerankModelCatalogService:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def build(self) -> RerankModelCatalogResponse:
        engine_supports = registry.active().capabilities.rerank
        settings = DefaultModelSettings(self._db)
        installed = set(await settings.installed_rerankers()) if engine_supports else set()
        in_use = await settings.rerank() if engine_supports else None
        entries = [
            RerankCatalogEntry(
                family=entry.family,
                vendor=entry.vendor,
                tag=entry.tag,
                parameter_size=entry.parameter_size,
                context_length=entry.context_length,
                download_gb=entry.download_gb,
                min_ram_gb=entry.min_ram_gb,
                installed=entry.tag in installed,
                in_use=entry.tag == in_use,
            )
            for entry in CATALOG.rerank_models
        ]
        return RerankModelCatalogResponse(engine_supports=engine_supports, entries=entries)
