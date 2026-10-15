"""Engine-agnostic front door for chat/embedding/model-management calls. Every call site outside the
Ollama/Matricxon admin plumbing (routers/settings.py, users.py; document_retrieval/sync/ingest,
reply_generation_service, reply_termination_service, chat_prompt_service, title_service,
model_catalog_service, extended_model_catalog_service, startup_service) imports from here instead of
ollama_client/matricxon_client. engines.registry resolves engine_service.current_engine per call, so an
engine switch (routers/engine_admin.py) applies to the next request.

Each function dispatches to the active InferenceEngine's method of the same name and normalizes every
engine's exception types into one InferenceError (one except clause for callers). A third engine needs no
change here.
"""

from collections.abc import AsyncGenerator

from app.services.engines.base import EngineCapabilities
from app.services.engines.registry import registry


class InferenceError(Exception):
    """Raised when the active engine is unreachable or errors; wraps its error types (InferenceEngine.error_types)
    or EngineCapabilityError, so callers need no engine-specific import.
    """


def active_capabilities() -> EngineCapabilities:
    """What the active engine can do, for callers that check before calling (e.g. Settings greying out the RAG
    embedding picker when it can't embed) rather than catching InferenceError.
    """
    return registry.active().capabilities


async def list_models() -> list[dict]:
    try:
        return await registry.active().list_models()
    except registry.all_error_types() as exc:
        raise InferenceError(str(exc)) from exc


async def embed(text: str, model: str) -> list[float]:
    try:
        return await registry.active().embed(text, model)
    except registry.all_error_types() as exc:
        raise InferenceError(str(exc)) from exc


async def rerank(query: str, documents: list[str], model: str) -> list[float]:
    """Each document's relevance to `query` (input order); InferenceError if the engine can't rerank."""
    engine = registry.active()
    if not engine.capabilities.rerank:
        raise InferenceError(f"{engine.display_name} does not support reranking.")
    try:
        return await engine.rerank(query, documents, model)
    except registry.all_error_types() as exc:
        raise InferenceError(str(exc)) from exc


async def chat_stream(model: str, messages: list[dict], params: dict) -> AsyncGenerator[str, None]:
    try:
        async for chunk in registry.active().chat_stream(model, messages, params):
            yield chunk
    except registry.all_error_types() as exc:
        raise InferenceError(str(exc)) from exc


async def chat_once(model: str, messages: list[dict], params: dict | None = None) -> str:
    try:
        return await registry.active().chat_once(model, messages, params)
    except registry.all_error_types() as exc:
        raise InferenceError(str(exc)) from exc


async def stop_model(model: str, request_id: str | None = None) -> None:
    await registry.active().stop_model(model, request_id)


async def pull_model_stream(tag: str):
    """Every pulled model, for any engine, must be an explicit hf.co/<repo>:<file> tag, never a bare Ollama-library
    name ("llama3"). The same string then names the same file on every engine, and no model only one engine can
    resolve: Matricxon can't use Ollama's registry (see matricxon_installer), so such a tag would break on an
    engine switch (engine_service). Settings > Model's Pull already offers only hf.co: tags (model_catalog.py,
    ExtendedModelCatalog.add); this backstops direct callers. App-level policy, whatever the engine.
    """
    if not tag.startswith("hf.co/"):
        raise InferenceError(f'"{tag}" is not an hf.co/<repo>:<file> tag — only Hugging Face model tags can be pulled.')
    try:
        async for progress in registry.active().pull_model_stream(tag):
            yield progress
    except registry.all_error_types() as exc:
        raise InferenceError(str(exc)) from exc


async def delete_model(tag: str) -> None:
    try:
        await registry.active().delete_model(tag)
    except registry.all_error_types() as exc:
        raise InferenceError(str(exc)) from exc
