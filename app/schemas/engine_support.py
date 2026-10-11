from pydantic import BaseModel


class EngineSupportResponse(BaseModel):
    """GET /api/stats/engine-support: the Stats "Supported architectures" view
    (engine_support_service.EngineSupportService).
    Only Matricxon exposes it (GET /api/health), so `available` is False for Ollama. `error` is set (lists empty)
    if Matricxon can't be reached.
    """

    active_engine: str
    available: bool
    architectures: list[str] = []
    quantizations: list[str] = []
    moe_architectures: list[str] = []
    vision_architectures: list[str] = []
    error: str | None = None
