"""
The active engine's own supported model architectures and quantization (dequant) types, for the Stats page's
"Supported architectures" view. Matricxon-only: it's the one engine that exposes this (GET /api/health, see
app.services.matricxon_client.get_capabilities) — Ollama has no API listing what it can load, so an Ollama-active
response is simply `available=False` rather than a hand-maintained list that would silently drift.
"""

from app.schemas.engine_support import EngineSupportResponse
from app.services import engine_service, matricxon_client
from app.services.matricxon_client import MatricxonError


class EngineSupportService:
    @staticmethod
    async def get() -> EngineSupportResponse:
        active_engine = engine_service.current_engine()
        if active_engine != "matricxon":
            return EngineSupportResponse(active_engine=active_engine, available=False)
        try:
            capabilities = await matricxon_client.get_capabilities()
        except MatricxonError as exc:
            return EngineSupportResponse(active_engine=active_engine, available=True, error=str(exc))
        return EngineSupportResponse(
            active_engine=active_engine,
            available=True,
            architectures=capabilities.get("supported_architectures", []),
            quantizations=capabilities.get("supported_quantizations", []),
        )
