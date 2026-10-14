"""GroqEngine: InferenceEngine adapter over groq_client. Forwards by module reference, looked up per call
(monkeypatch-testable, see base.py); its key comes from connector_config_cache."""

from collections.abc import AsyncGenerator

from app.services import connector_config_cache, groq_client
from app.services.engines.base import EngineCapabilities, InferenceEngine
from app.services.groq_client import GroqError

_USAGE_NOTE = (
    "Groq has no API for usage: see your rate limits and usage at console.groq.com (Settings > Limits, and Usage)."
)


class GroqEngine(InferenceEngine):
    name = "groq"
    display_name = "Groq"
    capabilities = EngineCapabilities(
        embeddings=False,
        model_management=False,
        stop_model=False,
        local_process=False,
        host_pool=False,
        support_checking=False,
        format_introspection=False,
    )
    error_types = (GroqError,)

    async def list_models(self) -> list[dict]:
        return await groq_client.list_models()

    def chat_stream(self, model: str, messages: list[dict], params: dict) -> AsyncGenerator[str, None]:
        return groq_client.chat_stream(model, messages, params)

    async def chat_once(self, model: str, messages: list[dict], params: dict | None = None) -> str:
        return await groq_client.chat_once(model, messages, params)

    async def check_health(self) -> dict[str, bool]:
        return await groq_client.check_health()

    async def usage_summary(self) -> tuple[dict | None, str | None]:
        return None, _USAGE_NOTE

    async def is_ready(self) -> bool:
        """Ready once saved, enabled and "Test connection" passed (connector_config_cache)."""
        return connector_config_cache.is_ready("groq")

    async def test_connection(self, config: dict[str, str]) -> tuple[bool, str]:
        return await groq_client.test_connection(config)
