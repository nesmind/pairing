"""Unit tests for app/services/engine_service.py — persistence + the in-process cache current_engine() reads on
every hot-path inference_client call."""

import pytest

from app.services import engine_service


@pytest.fixture(autouse=True)
def reset_cache():
    """Each test gets a clean in-process cache — engine_service._cached_engine is module-level state that would
    otherwise leak between tests."""
    engine_service._cached_engine = engine_service.DEFAULT_ENGINE
    yield
    engine_service._cached_engine = engine_service.DEFAULT_ENGINE


@pytest.mark.asyncio
async def test_get_active_engine_defaults_when_never_configured(db):
    assert await engine_service.get_active_engine(db) == engine_service.DEFAULT_ENGINE


@pytest.mark.asyncio
async def test_set_active_engine_persists_and_is_read_back(db):
    await engine_service.set_active_engine(db, "ollama")
    assert await engine_service.get_active_engine(db) == "ollama"


@pytest.mark.asyncio
async def test_set_active_engine_updates_the_in_process_cache_immediately(db):
    assert engine_service.current_engine() == engine_service.DEFAULT_ENGINE
    await engine_service.set_active_engine(db, "ollama")
    assert engine_service.current_engine() == "ollama"


@pytest.mark.asyncio
async def test_load_cache_from_db_populates_the_cache_from_a_saved_value(db):
    await engine_service.set_active_engine(db, "ollama")
    engine_service._cached_engine = "matricxon"  # simulate a sibling instance that hasn't refreshed yet
    await engine_service.load_cache_from_db(db)
    assert engine_service.current_engine() == "ollama"


@pytest.mark.asyncio
async def test_load_cache_from_db_defaults_when_never_configured(db):
    engine_service._cached_engine = "ollama"
    await engine_service.load_cache_from_db(db)
    assert engine_service.current_engine() == engine_service.DEFAULT_ENGINE
