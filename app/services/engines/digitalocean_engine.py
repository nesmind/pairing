"""DigitalOceanEngine: InferenceEngine adapter over digitalocean_client. Forwards by module reference, looked up per
call (monkeypatch-testable, see base.py); its key comes from connector_config_cache."""

from collections.abc import AsyncGenerator

from app.services import connector_config_cache, digitalocean_client
from app.services.digitalocean_client import DigitalOceanError
from app.services.engines.base import EngineCapabilities, InferenceEngine

_USAGE_NOTE = (
    "DigitalOcean has no usage API for Serverless Inference: see usage and costs in its control panel "
    "(Billing, or Inference Engine > Serverless Inference)."
)


class DigitalOceanEngine(InferenceEngine):
    name = "digitalocean"
    display_name = "DigitalOcean Serverless Inference"
    capabilities = EngineCapabilities(
        embeddings=True,
        model_management=False,
        stop_model=False,
        local_process=False,
        host_pool=False,
        support_checking=False,
        format_introspection=False,
    )
    error_types = (DigitalOceanError,)

    async def list_models(self) -> list[dict]:
        return await digitalocean_client.list_models()

    def chat_stream(self, model: str, messages: list[dict], params: dict) -> AsyncGenerator[str, None]:
        return digitalocean_client.chat_stream(model, messages, params)

    async def chat_once(self, model: str, messages: list[dict], params: dict | None = None) -> str:
        return await digitalocean_client.chat_once(model, messages, params)

    async def embed(self, text: str, model: str) -> list[float]:
        return await digitalocean_client.embed(text, model)

    async def check_health(self) -> dict[str, bool]:
        return await digitalocean_client.check_health()

    async def account_details(self) -> list[tuple[str, str]]:
        return await digitalocean_client.rate_limit_details()

    async def usage_summary(self) -> tuple[dict | None, str | None]:
        return None, _USAGE_NOTE

    async def is_ready(self) -> bool:
        """Ready once saved, enabled and "Test connection" passed (connector_config_cache)."""
        return connector_config_cache.is_ready("digitalocean")

    async def test_connection(self, config: dict[str, str]) -> tuple[bool, str]:
        return await digitalocean_client.test_connection(config)
