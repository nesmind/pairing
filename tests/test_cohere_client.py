"""Tests for services/cohere_client.py (Cohere through its OpenAI-compatible API; models from Cohere's own list).
httpx.AsyncClient gets a MockTransport: no network."""

import json

import httpx
import pytest

from app.services import cohere_client, connector_config_cache
from app.services.cohere_client import CohereError
from app.services.engines.cohere_engine import CohereEngine

PARAMS = {"temperature": 0.5, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1, "num_predict": 64, "seed": 7}
MODELS = [
    {"name": "command-a-03-2025", "endpoints": ["chat"], "context_length": 256000, "is_deprecated": False},
    {"name": "embed-v4.0", "endpoints": ["embed"], "context_length": 128000},
    {"name": "rerank-v3.5", "endpoints": ["rerank"]},
    {"name": "command-light", "endpoints": ["generate", "chat"], "is_deprecated": True},
]


@pytest.fixture(autouse=True)
def ready():
    connector_config_cache._cache = {
        "cohere": {"config": {"api_key": "k"}, "enabled": True, "configured": True, "last_test_passed": True}
    }
    yield
    connector_config_cache._cache = {}


def _fake_api(monkeypatch, handler):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))


@pytest.mark.asyncio
async def test_lists_chat_and_embed_models_from_cohere_s_own_list(monkeypatch):
    def handler(request):
        assert str(request.url) == "https://api.cohere.com/v1/models?page_size=1000"
        return httpx.Response(200, json={"models": MODELS})

    _fake_api(monkeypatch, handler)
    models = {m["name"]: m for m in await cohere_client.PROVIDER.list_models()}

    assert {n: m["capabilities"] for n, m in models.items()} == {
        "command-a-03-2025": ["completion"],
        "embed-v4.0": ["embedding"],
    }  # no rerank-only or deprecated models
    assert models["command-a-03-2025"]["vendor"] == "Cohere"
    assert models["command-a-03-2025"]["details"] == {"context_length": 256000}


@pytest.mark.asyncio
async def test_chat_goes_to_the_compatibility_api_with_supported_fields(monkeypatch):
    seen = {}

    def handler(request):
        seen["url"], seen["body"] = str(request.url), json.loads(request.content)
        sse = 'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\ndata: [DONE]\n\n'
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    _fake_api(monkeypatch, handler)
    reply = await cohere_client.PROVIDER.chat_once("command-a-03-2025", [{"role": "user", "content": "hi"}], PARAMS)

    assert reply == "Hi"
    assert seen["url"] == "https://api.cohere.ai/compatibility/v1/chat/completions"
    assert set(seen["body"]) == {"model", "messages", "stream", "temperature", "top_p", "seed"}


@pytest.mark.asyncio
async def test_embed_asks_for_floats(monkeypatch):
    seen = {}

    def handler(request):
        seen["url"], seen["body"] = str(request.url), json.loads(request.content)
        return httpx.Response(200, json={"data": [{"embedding": [0.5, 0.25]}]})

    _fake_api(monkeypatch, handler)
    assert await CohereEngine().embed("hello", "embed-v4.0") == [0.5, 0.25]
    assert seen["url"] == "https://api.cohere.ai/compatibility/v1/embeddings"
    assert seen["body"] == {"model": "embed-v4.0", "input": "hello", "encoding_format": "float"}


@pytest.mark.asyncio
async def test_the_trial_limit_gets_a_clear_message(monkeypatch):
    _fake_api(monkeypatch, lambda r: httpx.Response(429, json={"message": "trial limit"}))
    with pytest.raises(CohereError, match="trial key allows about 1,000 calls a month"):
        await cohere_client.PROVIDER.chat_once("command-a-03-2025", [{"role": "user", "content": "hi"}], PARAMS)


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "ok", "fragment"), [(200, True, "Connected"), (401, False, "dashboard.cohere.com")])
async def test_test_connection_uses_the_model_list(monkeypatch, status, ok, fragment):
    _fake_api(monkeypatch, lambda r: httpx.Response(status, json={"models": []}))
    passed, message = await cohere_client.PROVIDER.test_connection({"api_key": "k"})
    assert passed is ok and fragment in message


@pytest.mark.asyncio
async def test_usage_points_to_the_dashboard():
    assert "Cohere dashboard" in (await CohereEngine().usage_summary())[1]
