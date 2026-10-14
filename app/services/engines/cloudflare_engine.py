"""CloudflareEngine: InferenceEngine adapter over cloudflare_client. Forwards by module reference, looked up
per call (monkeypatch-testable, see base.py); its config comes from connector_config_cache."""

from collections.abc import AsyncGenerator

from app.services import cloudflare_client, connector_config_cache
from app.services.cloudflare_client import CloudflareError
from app.services.cloudflare_usage import CloudflareAccountReport
from app.services.engines.base import EngineCapabilities, InferenceEngine


class CloudflareEngine(InferenceEngine):
    name = "cloudflare"
    display_name = "Cloudflare Workers AI"
    capabilities = EngineCapabilities(
        embeddings=True,
        model_management=False,
        stop_model=False,
        local_process=False,
        host_pool=False,
        support_checking=False,
        format_introspection=False,
    )
    error_types = (CloudflareError,)

    async def list_models(self) -> list[dict]:
        return await cloudflare_client.list_models()

    def chat_stream(self, model: str, messages: list[dict], params: dict) -> AsyncGenerator[str, None]:
        return cloudflare_client.chat_stream(model, messages, params)

    async def chat_once(self, model: str, messages: list[dict], params: dict | None = None) -> str:
        return await cloudflare_client.chat_once(model, messages, params)

    async def embed(self, text: str, model: str) -> list[float]:
        return await cloudflare_client.embed(text, model)

    async def check_health(self) -> dict[str, bool]:
        return await cloudflare_client.check_health()

    async def account_details(self) -> list[tuple[str, str]]:
        return await CloudflareAccountReport().account_details()

    async def usage_summary(self) -> tuple[dict | None, str | None]:
        return await CloudflareAccountReport().usage_summary()

    async def is_ready(self) -> bool:
        """Ready once saved, enabled and "Test connection" passed (connector_config_cache)."""
        return connector_config_cache.is_ready("cloudflare")

    async def test_connection(self, config: dict[str, str]) -> tuple[bool, str]:
        return await cloudflare_client.test_connection(config)
