from pydantic import BaseModel


class EngineSupportResponse(BaseModel):
    """GET /api/stats/engine-support — the Stats page's "Supported architectures" view (see
    app.services.engine_support_service.EngineSupportService). Only Matricxon exposes this at all (its own
    GET /api/health); Ollama has no equivalent API, so `available` is False whenever Ollama is the active engine.
    `error` is set (and both lists empty) when Matricxon is active but couldn't be reached right now."""

    active_engine: str
    available: bool
    architectures: list[str] = []
    quantizations: list[str] = []
    error: str | None = None
