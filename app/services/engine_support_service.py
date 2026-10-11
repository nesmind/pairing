"""The active engine's supported architectures and quantization (dequant) types, for the Stats "Supported
architectures" view. Only for engines with format introspection (InferenceEngine.capabilities.format_introspection:
Matricxon via GET /api/health, matricxon_client.get_capabilities). Ollama has no such API, so it returns
`available=False` rather than a drifting hand-kept list; so would any future engine without it.
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
            vision_architectures=capabilities.get("vision_supported_architectures", []),
        )
