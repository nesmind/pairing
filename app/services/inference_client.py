"""
Engine-agnostic front door for chat/embedding/model-management calls — every call site outside the Ollama- and
Matricxon-specific admin plumbing itself (routers/settings.py, users.py; services/document_retrieval.py,
document_sync.py, document_ingest.py, reply_generation_service.py, reply_termination_service.py,
chat_prompt_service.py, title_service.py, model_catalog_service.py, extended_model_catalog_service.py,
startup_service.py) imports from here instead of reaching into app.services.ollama_client or
app.services.matricxon_client directly, so none of them need to know or care which engine is actually active —
see app.services.engines.registry, which resolves app.services.engine_service.current_engine() fresh on every
call so a mid-session engine switch (see app/routers/engine_admin.py) takes effect on the very next request, not
just after a restart.

Each function is a thin dispatch to the active app.services.engines.base.InferenceEngine's own method of the
same name — this file adds no behavior of its own beyond that routing and normalizing every registered engine's
own exception type(s) into one InferenceError, so a caller only ever needs one except clause regardless of which
engine served (or failed to serve) its request. Adding a third engine (see app.services.engines.registry's own
docstring) needs no change here at all."""

from collections.abc import AsyncGenerator

from app.services.engines.base import EngineCapabilities
from app.services.engines.registry import registry


class InferenceError(Exception):
    """Raised whenever the currently-active engine is unreachable or returns an error — wraps whichever of the
    active engine's own error types (see InferenceEngine.error_types) or EngineCapabilityError it actually
    raised, so callers never need to know or import any engine-specific type."""


def active_capabilities() -> EngineCapabilities:
    """What the currently-active engine can actually do — for a caller that needs to check before calling (e.g.
    Settings greying out the RAG embedding model picker when the active engine can't embed) rather than after,
    via a caught InferenceError."""
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
    """Every model this app pulls — for any engine — must be an explicit hf.co/<repo>:<file> tag, never a
    bare Ollama-library name (e.g. "llama3") or any other registry shorthand. Keeps a tag portable: the same
    string always names the same real file regardless of which engine ends up loading it, and never produces a
    model only one engine can ever resolve — Matricxon has no access to Ollama's own registry protocol at all
    (see app.services.matricxon_installer's own docstring), so a bare Ollama-library tag pulled there would be
    unusable the moment an admin switched the active engine (see app.services.engine_service). The app's own
    Settings > Model "Pull" button already only ever offers hf.co: tags (every entry in app/model_catalog.py and
    everything app.services.extended_model_catalog_service.ExtendedModelCatalog.add constructs is one) — this
    is the backstop for any caller that reaches this function directly instead. This is app-level policy, not
    an engine concern, so it applies regardless of which engine is active."""
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
