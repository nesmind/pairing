"""MatricxonEngine — the InferenceEngine adapter around app.services.matricxon_client/
matricxon_admin/matricxon_pool. Same "forward by module reference, looked up again on
every call" contract as OllamaEngine — see that module's own docstring."""

from collections.abc import AsyncGenerator

from app.services import matricxon_admin, matricxon_client, matricxon_pool
from app.services.engines.base import EngineCapabilities, InferenceEngine
from app.services.matricxon_client import MatricxonError
from app.services.matricxon_support_checker import MatricxonSupportChecker


class MatricxonEngine(InferenceEngine):
    name = "matricxon"
    display_name = "Matricxon"
    capabilities = EngineCapabilities(
        embeddings=True,
        model_management=True,
        stop_model=True,
        local_process=True,
        host_pool=True,
        support_checking=True,
        format_introspection=True,
    )
    error_types = (MatricxonError,)
    support_checker = MatricxonSupportChecker

    async def list_models(self) -> list[dict]:
        return await matricxon_client.list_models()

    def chat_stream(self, model: str, messages: list[dict], params: dict) -> AsyncGenerator[str, None]:
        return matricxon_client.chat_stream(model, messages, params)

    async def chat_once(self, model: str, messages: list[dict], params: dict | None = None) -> str:
        return await matricxon_client.chat_once(model, messages, params)

    async def check_health(self) -> dict[str, bool]:
        return await matricxon_pool.check_hosts()

    async def embed(self, text: str, model: str) -> list[float]:
        return await matricxon_client.embed(text, model)

    async def stop_model(self, model: str) -> None:
        await matricxon_client.stop_model(model)

    def pull_model_stream(self, tag: str) -> AsyncGenerator[dict, None]:
        return matricxon_admin.pull_model_stream(tag)

    async def delete_model(self, tag: str) -> None:
        await matricxon_admin.delete_model(tag)

    async def get_format_support(self) -> dict | None:
        return await matricxon_client.get_capabilities()
