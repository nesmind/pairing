"""RunPodEngine — the InferenceEngine adapter around app.services.runpod_client. Every
method forwards by module reference, looked up again on every call (never bound to a
local name), matching every other engine adapter's monkeypatch-testability
convention (see base.py's own docstring). No local process, no host pool — see
app.services.connectors and app.services.connector_config_cache for how this
engine's live config (endpoint ID, API key, model) reaches runpod_client without a
db session anywhere on this call path."""

from collections.abc import AsyncGenerator

from app.services import connector_config_cache, runpod_client
from app.services.engines.base import EngineCapabilities, InferenceEngine
from app.services.runpod_client import RunPodError


class RunPodEngine(InferenceEngine):
    name = "runpod"
    display_name = "RunPod Serverless"
    capabilities = EngineCapabilities(
        embeddings=False,
        model_management=False,
        stop_model=False,
        local_process=False,
        host_pool=False,
        support_checking=False,
        format_introspection=False,
    )
    error_types = (RunPodError,)

    async def list_models(self) -> list[dict]:
        return await runpod_client.list_models()

    def chat_stream(self, model: str, messages: list[dict], params: dict) -> AsyncGenerator[str, None]:
        return runpod_client.chat_stream(model, messages, params)

    async def chat_once(self, model: str, messages: list[dict], params: dict | None = None) -> str:
        return await runpod_client.chat_once(model, messages, params)

    async def check_health(self) -> dict[str, bool]:
        return await runpod_client.check_health()

    async def is_ready(self) -> bool:
        """Not ready until an admin has saved valid config for the RunPod connector, enabled it, and it's passed
        a "Test connection" check on the Connectors page (see app.services.connector_config_cache)."""
        return connector_config_cache.is_ready("runpod")

    async def test_connection(self, config: dict[str, str]) -> tuple[bool, str]:
        return await runpod_client.test_connection(config)
