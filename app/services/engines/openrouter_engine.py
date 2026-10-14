"""OpenRouterEngine: InferenceEngine adapter over openrouter_client. Forwards by module reference, looked up per call
(monkeypatch-testable, see base.py); its key comes from connector_config_cache."""

from collections.abc import AsyncGenerator

from app.services import connector_config_cache, openrouter_client
from app.services.engines.base import EngineCapabilities, InferenceEngine
from app.services.openrouter_client import OpenRouterError


class OpenRouterEngine(InferenceEngine):
    name = "openrouter"
    display_name = "OpenRouter"
    capabilities = EngineCapabilities(
        embeddings=True,
        model_management=False,
        stop_model=False,
        local_process=False,
        host_pool=False,
        support_checking=False,
        format_introspection=False,
    )
    error_types = (OpenRouterError,)

    async def list_models(self) -> list[dict]:
        return await openrouter_client.list_models()

    def chat_stream(self, model: str, messages: list[dict], params: dict) -> AsyncGenerator[str, None]:
        return openrouter_client.chat_stream(model, messages, params)

    async def chat_once(self, model: str, messages: list[dict], params: dict | None = None) -> str:
        return await openrouter_client.chat_once(model, messages, params)

    async def embed(self, text: str, model: str) -> list[float]:
        return await openrouter_client.embed(text, model)

    async def check_health(self) -> dict[str, bool]:
        return await openrouter_client.check_health()

    async def account_details(self) -> list[tuple[str, str]]:
        return await openrouter_client.account_details()

    async def is_ready(self) -> bool:
        """Ready once saved, enabled and "Test connection" passed (connector_config_cache)."""
        return connector_config_cache.is_ready("openrouter")

    async def test_connection(self, config: dict[str, str]) -> tuple[bool, str]:
        return await openrouter_client.test_connection(config)
