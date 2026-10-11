"""RunPodEngine: InferenceEngine adapter over runpod_client. Forwards by module reference, looked up per
call (monkeypatch-testable, see base.py). No local process or host pool; its live config (endpoint ID, API
key, model) reaches runpod_client without a db session via services.connectors / connector_config_cache.
"""

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
        """Not ready until an admin saved valid RunPod config, enabled it and passed "Test connection"
        (connector_config_cache)."""
        return connector_config_cache.is_ready("runpod")

    async def test_connection(self, config: dict[str, str]) -> tuple[bool, str]:
        return await runpod_client.test_connection(config)
