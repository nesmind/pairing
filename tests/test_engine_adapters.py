"""Unit tests for app/services/engines/ollama_engine.py, matricxon_engine.py, and runpod_engine.py — confirms each
method forwards to its client/admin/pool module by reference (looked up again on every call, never bound at
construction time), so the existing test suite's monkeypatch.setattr(ollama_client, ...)-style patches keep
working through the adapter layer exactly the way they already do through app.services.inference_client."""

import pytest

from app.services import (
    connector_config_cache,
    matricxon_admin,
    matricxon_client,
    matricxon_pool,
    ollama_admin,
    ollama_client,
    ollama_pool,
    runpod_client,
)
from app.services.engines.matricxon_engine import MatricxonEngine
from app.services.engines.ollama_engine import OllamaEngine
from app.services.engines.runpod_engine import RunPodEngine


@pytest.fixture(autouse=True)
def reset_connector_cache():
    connector_config_cache._cache = {}
    yield
    connector_config_cache._cache = {}


@pytest.mark.asyncio
async def test_ollama_engine_list_models_forwards_to_ollama_client(monkeypatch):
    async def fake():
        return [{"name": "m"}]

    monkeypatch.setattr(ollama_client, "list_models", fake)
    assert await OllamaEngine().list_models() == [{"name": "m"}]


@pytest.mark.asyncio
async def test_ollama_engine_check_health_forwards_to_ollama_pool(monkeypatch):
    async def fake():
        return {"http://h1": True}

    monkeypatch.setattr(ollama_pool, "check_hosts", fake)
    assert await OllamaEngine().check_health() == {"http://h1": True}


@pytest.mark.asyncio
async def test_ollama_engine_delete_model_forwards_to_ollama_admin(monkeypatch):
    calls = []

    async def fake(tag):
        calls.append(tag)

    monkeypatch.setattr(ollama_admin, "delete_model", fake)
    await OllamaEngine().delete_model("some-tag")
    assert calls == ["some-tag"]


@pytest.mark.asyncio
async def test_matricxon_engine_list_models_forwards_to_matricxon_client(monkeypatch):
    async def fake():
        return [{"name": "m"}]

    monkeypatch.setattr(matricxon_client, "list_models", fake)
    assert await MatricxonEngine().list_models() == [{"name": "m"}]


@pytest.mark.asyncio
async def test_matricxon_engine_check_health_forwards_to_matricxon_pool(monkeypatch):
    async def fake():
        return {"http://m1": True}

    monkeypatch.setattr(matricxon_pool, "check_hosts", fake)
    assert await MatricxonEngine().check_health() == {"http://m1": True}


@pytest.mark.asyncio
async def test_matricxon_engine_get_format_support_forwards_to_matricxon_client(monkeypatch):
    async def fake():
        return {"supported_architectures": ["mistral3"]}

    monkeypatch.setattr(matricxon_client, "get_capabilities", fake)
    assert await MatricxonEngine().get_format_support() == {"supported_architectures": ["mistral3"]}


@pytest.mark.asyncio
async def test_matricxon_engine_delete_model_forwards_to_matricxon_admin(monkeypatch):
    calls = []

    async def fake(tag):
        calls.append(tag)

    monkeypatch.setattr(matricxon_admin, "delete_model", fake)
    await MatricxonEngine().delete_model("some-tag")
    assert calls == ["some-tag"]


def test_ollama_engine_has_no_format_introspection():
    assert OllamaEngine().capabilities.format_introspection is False


def test_matricxon_engine_has_format_introspection():
    assert MatricxonEngine().capabilities.format_introspection is True


@pytest.mark.asyncio
async def test_ollama_engine_is_ready_defaults_true():
    assert await OllamaEngine().is_ready() is True


@pytest.mark.asyncio
async def test_matricxon_engine_is_ready_defaults_true():
    assert await MatricxonEngine().is_ready() is True


@pytest.mark.asyncio
async def test_runpod_engine_list_models_forwards_to_runpod_client(monkeypatch):
    async def fake():
        return [{"name": "m"}]

    monkeypatch.setattr(runpod_client, "list_models", fake)
    assert await RunPodEngine().list_models() == [{"name": "m"}]


@pytest.mark.asyncio
async def test_runpod_engine_check_health_forwards_to_runpod_client(monkeypatch):
    async def fake():
        return {"runpod": True}

    monkeypatch.setattr(runpod_client, "check_health", fake)
    assert await RunPodEngine().check_health() == {"runpod": True}


def test_runpod_engine_has_no_embeddings_or_local_process():
    capabilities = RunPodEngine().capabilities
    assert capabilities.embeddings is False
    assert capabilities.local_process is False
    assert capabilities.host_pool is False


@pytest.mark.asyncio
async def test_runpod_engine_is_ready_reflects_the_connector_cache():
    assert await RunPodEngine().is_ready() is False

    connector_config_cache._cache = {
        "runpod": {"config": {}, "enabled": True, "configured": True, "last_test_passed": True}
    }
    assert await RunPodEngine().is_ready() is True


@pytest.mark.asyncio
async def test_runpod_engine_is_ready_false_without_a_passed_test():
    connector_config_cache._cache = {
        "runpod": {"config": {}, "enabled": True, "configured": True, "last_test_passed": False}
    }
    assert await RunPodEngine().is_ready() is False


@pytest.mark.asyncio
async def test_runpod_engine_test_connection_forwards_to_runpod_client(monkeypatch):
    async def fake(config):
        assert config == {"endpoint_id": "ep-1"}
        return True, "ok"

    monkeypatch.setattr(runpod_client, "test_connection", fake)
    assert await RunPodEngine().test_connection({"endpoint_id": "ep-1"}) == (True, "ok")
