"""
The active engine's own supported model architectures and quantization (dequant) types, for the Stats page's
"Supported architectures" view. Only meaningful for an engine with real format introspection (see
InferenceEngine.capabilities.format_introspection — Matricxon today, via GET /api/health, see
app.services.matricxon_client.get_capabilities) — Ollama has no API listing what it can load, so an Ollama-active
response is simply `available=False` rather than a hand-maintained list that would silently drift. A future
engine without this capability gets the same honest `available=False`, with no code change needed here.
"""

from app.schemas.engine_support import EngineSupportResponse
from app.services.engines.registry import registry


class EngineSupportService:
    @staticmethod
    async def get() -> EngineSupportResponse:
        engine = registry.active()
        if not engine.capabilities.format_introspection:
            return EngineSupportResponse(active_engine=engine.name, available=False)
        try:
            capabilities = await engine.get_format_support()
        except engine.error_types as exc:
            return EngineSupportResponse(active_engine=engine.name, available=True, error=str(exc))
        return EngineSupportResponse(
            active_engine=engine.name,
            available=True,
            architectures=capabilities.get("supported_architectures", []),
            quantizations=capabilities.get("supported_quantizations", []),
            moe_architectures=capabilities.get("moe_supported_architectures", []),
        )
