"""Admin settings: the Knowledge base reranker."""

from pydantic import BaseModel


class RerankModelSetting(BaseModel):
    # The reranker in use; None when off or none is installed.
    model: str | None
    enabled: bool
    installed: list[str]
    # Whether the active engine can rerank at all.
    engine_supports: bool


class RerankModelUpdate(BaseModel):
    # None switches reranking off.
    model: str | None


class RerankCatalogEntry(BaseModel):
    family: str
    vendor: str
    tag: str
    parameter_size: str
    context_length: int | None
    download_gb: float | None
    min_ram_gb: float
    installed: bool
    # The reranker Knowledge base retrieval uses now.
    in_use: bool


class RerankModelCatalogResponse(BaseModel):
    engine_supports: bool
    entries: list[RerankCatalogEntry]
