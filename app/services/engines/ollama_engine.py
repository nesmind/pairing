"""OllamaEngine — the InferenceEngine adapter around app.services.ollama_client/
ollama_admin/ollama_pool. Every method forwards by module reference, looked up again
on every call (never bound to a local name), so the existing test suite's
`monkeypatch.setattr(ollama_client, "list_models", ...)`-style patches keep working
unchanged."""

from collections.abc import AsyncGenerator

from app.services import ollama_admin, ollama_client, ollama_pool
from app.services.engine_support_checker import OllamaSupportChecker
from app.services.engines.base import EngineCapabilities, InferenceEngine
from app.services.ollama_client import OllamaError


class OllamaEngine(InferenceEngine):
    name = "ollama"
    display_name = "Ollama"
    capabilities = EngineCapabilities(
        embeddings=True,
        model_management=True,
        stop_model=True,
        local_process=True,
        host_pool=True,
        support_checking=False,
        format_introspection=False,
    )
    error_types = (OllamaError,)
    support_checker = OllamaSupportChecker

    async def list_models(self) -> list[dict]:
        return await ollama_client.list_models()

    def chat_stream(self, model: str, messages: list[dict], params: dict) -> AsyncGenerator[str, None]:
        return ollama_client.chat_stream(model, messages, params)

    async def chat_once(self, model: str, messages: list[dict], params: dict | None = None) -> str:
        return await ollama_client.chat_once(model, messages, params)

    async def check_health(self) -> dict[str, bool]:
        return await ollama_pool.check_hosts()

    async def embed(self, text: str, model: str) -> list[float]:
        return await ollama_client.embed(text, model)

    async def stop_model(self, model: str) -> None:
        await ollama_client.stop_model(model)

    def pull_model_stream(self, tag: str) -> AsyncGenerator[dict, None]:
        return ollama_admin.pull_model_stream(tag)

    async def delete_model(self, tag: str) -> None:
        await ollama_admin.delete_model(tag)
