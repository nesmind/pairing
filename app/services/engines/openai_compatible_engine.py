"""InferenceEngine adapter for a provider built on openai_compatible.OpenAICompatibleProvider: every call forwards to
the class's `provider` (patchable in tests, like the module references of the other adapters)."""

from collections.abc import AsyncGenerator
from typing import ClassVar

from app.services import connector_config_cache
from app.services.engines.base import EngineCapabilities, InferenceEngine
from app.services.openai_compatible import OpenAICompatibleProvider

# A cloud connector: no local process, model downloads or host pool; text only for now.
CLOUD_CAPABILITIES = dict(
    model_management=False,
    stop_model=False,
    local_process=False,
    host_pool=False,
    support_checking=False,
    format_introspection=False,
)


def cloud_capabilities(*, embeddings: bool) -> EngineCapabilities:
    return EngineCapabilities(embeddings=embeddings, **CLOUD_CAPABILITIES)


class OpenAICompatibleEngine(InferenceEngine):
    provider: ClassVar[OpenAICompatibleProvider]
    usage_note: ClassVar[str | None] = None  # shown in External servers when the provider has no usage API

    async def list_models(self) -> list[dict]:
        return await self.provider.list_models()

    def chat_stream(self, model: str, messages: list[dict], params: dict) -> AsyncGenerator[str, None]:
        return self.provider.chat_stream(model, messages, params)

    async def chat_once(self, model: str, messages: list[dict], params: dict | None = None) -> str:
        return await self.provider.chat_once(model, messages, params)

    async def embed(self, text: str, model: str) -> list[float]:
        if not self.capabilities.embeddings:
            return await super().embed(text, model)
        return await self.provider.embed(text, model)

    async def check_health(self) -> dict[str, bool]:
        return await self.provider.check_health()

    async def usage_summary(self) -> tuple[dict | None, str | None]:
        return None, self.usage_note

    async def is_ready(self) -> bool:
        """Ready once saved, enabled and "Test connection" passed (connector_config_cache)."""
        return connector_config_cache.is_ready(self.provider.connector_id)

    async def test_connection(self, config: dict[str, str]) -> tuple[bool, str]:
        return await self.provider.test_connection(config)
