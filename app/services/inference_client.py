"""
Engine-agnostic front door for chat/embedding/model-management calls — every call site outside the Ollama- and
Matricxon-specific admin plumbing itself (routers/settings.py, users.py; services/document_retrieval.py,
document_sync.py, document_ingest.py, reply_generation_service.py, reply_termination_service.py,
chat_prompt_service.py, title_service.py, model_catalog_service.py, extended_model_catalog_service.py,
startup_service.py) imports from here instead of reaching into app.services.ollama_client or
app.services.matricxon_client directly, so none of them need to know or care which engine is actually active —
see app.services.engine_service.current_engine, read fresh on every call so a mid-session engine switch (see
app/routers/engine_admin.py) takes effect on the very next request, not just after a restart.

Each function is a thin dispatch to the matching app.services.ollama_client / app.services.matricxon_client
function of the same name — this file adds no behavior of its own beyond that routing and normalizing both
engines' own exception types into one InferenceError, so a caller only ever needs one except clause regardless
of which engine served (or failed to serve) its request.
"""

from collections.abc import AsyncGenerator

from app.services import engine_service, matricxon_admin, matricxon_client, ollama_admin, ollama_client
from app.services.matricxon_client import MatricxonError
from app.services.ollama_client import OllamaError


class InferenceError(Exception):
    """Raised whenever the currently-active engine is unreachable or returns an error — wraps
    app.services.ollama_client.OllamaError or app.services.matricxon_client.MatricxonError, whichever the active
    engine actually raised, so callers never need to know or import either engine-specific type."""


def _active_client():
    return matricxon_client if engine_service.current_engine() == "matricxon" else ollama_client


def _active_admin():
    return matricxon_admin if engine_service.current_engine() == "matricxon" else ollama_admin


async def list_models() -> list[dict]:
    try:
        return await _active_client().list_models()
    except (OllamaError, MatricxonError) as exc:
        raise InferenceError(str(exc)) from exc


async def embed(text: str, model: str) -> list[float]:
    try:
        return await _active_client().embed(text, model)
    except (OllamaError, MatricxonError) as exc:
        raise InferenceError(str(exc)) from exc


async def chat_stream(model: str, messages: list[dict], params: dict) -> AsyncGenerator[str, None]:
    try:
        async for chunk in _active_client().chat_stream(model, messages, params):
            yield chunk
    except (OllamaError, MatricxonError) as exc:
        raise InferenceError(str(exc)) from exc


async def chat_once(model: str, messages: list[dict], params: dict | None = None) -> str:
    try:
        return await _active_client().chat_once(model, messages, params)
    except (OllamaError, MatricxonError) as exc:
        raise InferenceError(str(exc)) from exc


async def stop_model(model: str) -> None:
    await _active_client().stop_model(model)


async def pull_model_stream(tag: str):
    """Every model this app pulls — for either engine — must be an explicit hf.co/<repo>:<file> tag, never a
    bare Ollama-library name (e.g. "llama3") or any other registry shorthand. Keeps a tag portable: the same
    string always names the same real file regardless of which engine ends up loading it, and never produces a
    model only one engine can ever resolve — Matricxon has no access to Ollama's own registry protocol at all
    (see app.services.matricxon_installer's own docstring), so a bare Ollama-library tag pulled there would be
    unusable the moment an admin switched the active engine (see app.services.engine_service). The app's own
    Settings > Model "Pull" button already only ever offers hf.co: tags (every entry in app/model_catalog.py and
    everything app.services.extended_model_catalog_service.ExtendedModelCatalog.add constructs is one) — this
    is the backstop for any caller that reaches this function directly instead."""
    if not tag.startswith("hf.co/"):
        raise InferenceError(f'"{tag}" is not an hf.co/<repo>:<file> tag — only Hugging Face model tags can be pulled.')
    try:
        async for progress in _active_admin().pull_model_stream(tag):
            yield progress
    except (OllamaError, MatricxonError) as exc:
        raise InferenceError(str(exc)) from exc


async def delete_model(tag: str) -> None:
    try:
        await _active_admin().delete_model(tag)
    except (OllamaError, MatricxonError) as exc:
        raise InferenceError(str(exc)) from exc
