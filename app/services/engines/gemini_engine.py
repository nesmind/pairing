"""GeminiEngine: InferenceEngine adapter over gemini_client. Forwards by module reference, looked up per call
(monkeypatch-testable, see base.py); its key comes from connector_config_cache."""

from collections.abc import AsyncGenerator

from app.services import connector_config_cache, gemini_client
from app.services.engines.base import EngineCapabilities, InferenceEngine
from app.services.gemini_client import GeminiError

_USAGE_NOTE = (
    "Google has no API for Gemini usage: see your rate limits and usage in Google AI Studio "
    "(aistudio.google.com > Usage). Free-tier prompts may be used by Google to improve its products."
)


class GeminiEngine(InferenceEngine):
    name = "gemini"
    display_name = "Google Gemini"
    capabilities = EngineCapabilities(
        embeddings=True,
        model_management=False,
        stop_model=False,
        local_process=False,
        host_pool=False,
        support_checking=False,
        format_introspection=False,
    )
    error_types = (GeminiError,)

    async def list_models(self) -> list[dict]:
        return await gemini_client.list_models()

    def chat_stream(self, model: str, messages: list[dict], params: dict) -> AsyncGenerator[str, None]:
        return gemini_client.chat_stream(model, messages, params)

    async def chat_once(self, model: str, messages: list[dict], params: dict | None = None) -> str:
        return await gemini_client.chat_once(model, messages, params)

    async def embed(self, text: str, model: str) -> list[float]:
        return await gemini_client.embed(text, model)

    async def check_health(self) -> dict[str, bool]:
        return await gemini_client.check_health()

    async def usage_summary(self) -> tuple[dict | None, str | None]:
        return None, _USAGE_NOTE

    async def is_ready(self) -> bool:
        """Ready once saved, enabled and "Test connection" passed (connector_config_cache)."""
        return connector_config_cache.is_ready("gemini")

    async def test_connection(self, config: dict[str, str]) -> tuple[bool, str]:
        return await gemini_client.test_connection(config)
