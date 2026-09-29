"""Unit tests for app/services/runpod_client.py — the RunPod Serverless
OpenAI-compatible HTTP calls. httpx itself is mocked (via respx-free monkeypatching
of httpx.AsyncClient) rather than hitting the real RunPod API — mirrors
tests/test_ollama_client.py's own "the HTTP boundary is the seam" convention."""

import httpx
import pytest

from app.services import connector_config_cache, runpod_client
from app.services.runpod_client import RunPodError

_READY_CONFIG = {"endpoint_id": "ep-1", "api_key": "k", "model": "fallback-model"}


@pytest.fixture(autouse=True)
def reset_cache():
    connector_config_cache._cache = {}
    yield
    connector_config_cache._cache = {}


def _mark_ready(config=None):
    connector_config_cache._cache = {
        "runpod": {
            "config": config or _READY_CONFIG,
            "enabled": True,
            "configured": True,
            "last_test_passed": True,
        }
    }


@pytest.mark.asyncio
async def test_list_models_raises_when_not_configured():
    with pytest.raises(RunPodError, match="isn't configured"):
        await runpod_client.list_models()


@pytest.mark.asyncio
async def test_list_models_maps_the_live_models_response(monkeypatch):
    _mark_ready()

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"id": "meta/llama-3"}]}

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, headers=None):
            assert url == "https://api.runpod.ai/v2/ep-1/openai/v1/models"
            assert headers == {"Authorization": "Bearer k"}
            return _FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: _FakeClient())

    assert await runpod_client.list_models() == [{"name": "meta/llama-3", "capabilities": ["completion"]}]


@pytest.mark.asyncio
async def test_list_models_falls_back_to_configured_model_on_http_error(monkeypatch):
    _mark_ready()

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, headers=None):
            raise httpx.ConnectError("boom")

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: _FakeClient())

    assert await runpod_client.list_models() == [{"name": "fallback-model", "capabilities": ["completion"]}]


@pytest.mark.asyncio
async def test_list_models_raises_when_call_fails_and_no_fallback_model_configured(monkeypatch):
    _mark_ready({"endpoint_id": "ep-1", "api_key": "k"})

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, headers=None):
            raise httpx.ConnectError("boom")

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: _FakeClient())

    with pytest.raises(RunPodError, match="Could not list"):
        await runpod_client.list_models()


@pytest.mark.asyncio
async def test_check_health_returns_false_immediately_when_not_configured():
    assert await runpod_client.check_health() == {"runpod": False}


@pytest.mark.asyncio
async def test_check_health_true_on_200(monkeypatch):
    _mark_ready()

    class _FakeResponse:
        status_code = 200

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, headers=None):
            assert url == "https://api.runpod.ai/v2/ep-1/health"
            return _FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: _FakeClient())

    assert await runpod_client.check_health() == {"runpod": True}


@pytest.mark.asyncio
async def test_check_health_false_on_network_error(monkeypatch):
    _mark_ready()

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, headers=None):
            raise httpx.ConnectError("boom")

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: _FakeClient())

    assert await runpod_client.check_health() == {"runpod": False}


@pytest.mark.asyncio
async def test_chat_once_raises_when_not_configured():
    with pytest.raises(RunPodError, match="isn't configured"):
        await runpod_client.chat_once("some-model", [{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_chat_stream_yields_content_chunks_and_stops_on_done(monkeypatch):
    _mark_ready()
    lines = [
        'data: {"choices": [{"delta": {"content": "Hel"}}]}',
        'data: {"choices": [{"delta": {"content": "lo"}}]}',
        "data: [DONE]",
    ]

    class _FakeStreamResponse:
        def raise_for_status(self):
            pass

        async def aiter_lines(self):
            for line in lines:
                yield line

    class _StreamCtx:
        async def __aenter__(self):
            return _FakeStreamResponse()

        async def __aexit__(self, *exc):
            return False

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def stream(self, method, url, json=None, headers=None):
            assert method == "POST"
            assert url == "https://api.runpod.ai/v2/ep-1/openai/v1/chat/completions"
            assert json["stream"] is True
            return _StreamCtx()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: _FakeClient())

    params = {"temperature": 0.5, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1, "num_predict": -1}
    chunks = [chunk async for chunk in runpod_client.chat_stream("m", [{"role": "user", "content": "hi"}], params)]

    assert "".join(chunks) == "Hello"


@pytest.mark.asyncio
async def test_test_connection_rejects_a_config_missing_required_fields():
    passed, message = await runpod_client.test_connection({"endpoint_id": "ep-1"})
    assert passed is False
    assert "required" in message


@pytest.mark.asyncio
async def test_test_connection_succeeds_on_200(monkeypatch):
    class _FakeResponse:
        status_code = 200

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, headers=None):
            assert url == "https://api.runpod.ai/v2/ep-1/health"
            assert headers == {"Authorization": "Bearer k"}
            return _FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: _FakeClient())

    passed, message = await runpod_client.test_connection({"endpoint_id": "ep-1", "api_key": "k"})

    assert passed is True
    assert message == "Connected successfully."


@pytest.mark.asyncio
async def test_test_connection_reports_a_bad_api_key(monkeypatch):
    class _FakeResponse:
        status_code = 401

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, headers=None):
            return _FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: _FakeClient())

    passed, message = await runpod_client.test_connection({"endpoint_id": "ep-1", "api_key": "wrong"})

    assert passed is False
    assert "API key" in message


@pytest.mark.asyncio
async def test_test_connection_reports_an_unknown_endpoint(monkeypatch):
    class _FakeResponse:
        status_code = 404

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, headers=None):
            return _FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: _FakeClient())

    passed, message = await runpod_client.test_connection({"endpoint_id": "nonexistent", "api_key": "k"})

    assert passed is False
    assert "not found" in message


@pytest.mark.asyncio
async def test_test_connection_reports_a_network_error(monkeypatch):
    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, headers=None):
            raise httpx.ConnectError("boom")

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: _FakeClient())

    passed, message = await runpod_client.test_connection({"endpoint_id": "ep-1", "api_key": "k"})

    assert passed is False
    assert "Could not reach" in message


@pytest.mark.asyncio
async def test_test_connection_does_not_require_a_ready_cache_entry():
    """Unlike list_models/chat_stream/check_health, test_connection takes the candidate config directly and
    must work even when connector_config_cache has nothing saved yet — that's the whole point of testing a
    config *before* it's allowed to become ready."""
    assert connector_config_cache.get_config("runpod") == {}
    passed, message = await runpod_client.test_connection({})
    assert passed is False
    assert "required" in message
