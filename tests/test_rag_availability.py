"""GET /api/settings/rag-availability (app/routers/settings.py) - checks the active engine's own
capabilities.embeddings before ever asking whether an embedding model tag is configured, so a
connector like RunPod (embeddings=False) gets an accurate, actionable reason instead of the
"no embedding model found" message, which would be true-but-misleading there (no embedding model
would ever fix it). Active engine set via engine_service._cached_engine directly, same pattern
tests/test_engine_support_service.py already uses.
"""

import pytest

from app.routers.settings import rag_availability
from app.schemas import RagLimits
from app.services import default_model_settings as default_model_settings_module, engine_service, settings_service
from app.services.model_catalog_service import set_default_embedding_model


async def _async_return(value):
    return value


@pytest.fixture(autouse=True)
def _reset_engine_cache():
    engine_service._cached_engine = engine_service.DEFAULT_ENGINE
    yield
    engine_service._cached_engine = engine_service.DEFAULT_ENGINE


@pytest.mark.asyncio
async def test_unavailable_with_no_embedding_model_configured_on_an_embeddings_capable_engine(db, user):
    engine_service._cached_engine = "ollama"

    result = await rag_availability(db=db, _user=user)

    assert result.available is False
    assert result.reason == "No embedding model found yet — see Settings > Model to install one."


@pytest.mark.asyncio
async def test_available_once_an_embedding_model_is_configured(db, user, monkeypatch):
    engine_service._cached_engine = "ollama"
    monkeypatch.setattr(
        default_model_settings_module,
        "list_models",
        lambda: _async_return([{"name": "nomic-embed-text", "capabilities": ["embedding"]}]),
    )
    await set_default_embedding_model(db, "nomic-embed-text")

    result = await rag_availability(db=db, _user=user)

    assert result.available is True
    assert result.reason is None


@pytest.mark.asyncio
async def test_unavailable_when_admin_disabled_rag_regardless_of_engine_or_configured_model(db, user, monkeypatch):
    """The admin on/off switch (RagLimits.enabled, Settings > System) is checked before anything else - an
    embeddings-capable engine with a real model configured must still come back unavailable once this is
    off, with the exact message Settings > Knowledge shows on its page title."""
    engine_service._cached_engine = "ollama"
    monkeypatch.setattr(
        default_model_settings_module,
        "list_models",
        lambda: _async_return([{"name": "nomic-embed-text", "capabilities": ["embedding"]}]),
    )
    await set_default_embedding_model(db, "nomic-embed-text")
    await settings_service.set_rag_limits(db, RagLimits(enabled=False, max_file_mb=10, max_user_space_mb=100))

    result = await rag_availability(db=db, _user=user)

    assert result.available is False
    assert result.reason == "RAG is not enabled on this system, ask admin"


@pytest.mark.asyncio
async def test_unavailable_on_an_engine_that_cannot_embed_regardless_of_any_configured_model(db, user):
    """The active engine's own capabilities.embeddings is checked first, before ever asking whether a model
    tag is configured (see rag_availability's own docstring) - no set_default_embedding_model call here at
    all, to prove this short-circuits rather than happening to pass because nothing was configured either."""
    engine_service._cached_engine = "runpod"

    result = await rag_availability(db=db, _user=user)

    assert result.available is False
    assert "RunPod Serverless" in result.reason
    assert "doesn't support embeddings" in result.reason
