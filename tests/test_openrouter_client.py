"""Tests for services/openrouter_client.py (OpenRouter's OpenAI-compatible API) and the mid-stream error handling in
services/openai_compat.py. httpx.AsyncClient gets a MockTransport: no network."""

import json

import httpx
import pytest

from app.services import connector_config_cache, openrouter_client
from app.services.engines.openrouter_engine import OpenRouterEngine
from app.services.openrouter_client import OpenRouterError

_PARAMS = {"temperature": 0.5, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1, "num_predict": -1, "seed": -1}
_CHAT = [
    {
        "id": "google/gemma-4-31b-it:free",
        "name": "Google: Gemma 4 31B (free)",
        "context_length": 262144,
        "pricing": {"prompt": "0", "completion": "0"},
    },
    {
        "id": "openai/gpt-5",
        "name": "OpenAI: GPT-5",
        "context_length": 400000,
        "pricing": {"prompt": "0.00000125", "completion": "0.00001"},
    },
]
_EMBEDDINGS = [
    {
        "id": "nvidia/nemotron-3-embed-1b:free",
        "name": "NVIDIA: Nemotron 3 Embed 1B (free)",
        "context_length": 8192,
        "pricing": {"prompt": "0", "completion": "0"},
    },
]


@pytest.fixture(autouse=True)
def reset_cache():
    connector_config_cache._cache = {}
    yield
    connector_config_cache._cache = {}


def _mark_ready(models: str = ""):
    config = {"api_key": "sk-or-key", "models": models}
    connector_config_cache._cache = {
        "openrouter": {"config": config, "enabled": True, "configured": True, "last_test_passed": True}
    }


def _fake_api(monkeypatch, handler):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))


def _catalog(request):
    assert request.headers["Authorization"] == "Bearer sk-or-key" and request.headers["X-Title"] == "pAIring"
    data = _EMBEDDINGS if request.url.path.endswith("/embeddings/models") else _CHAT
    return httpx.Response(200, json={"data": data})


@pytest.mark.asyncio
async def test_lists_only_free_models_by_default(monkeypatch):
    _mark_ready()
    _fake_api(monkeypatch, _catalog)
    models = await openrouter_client.list_models()
    assert [(m["name"], m["capabilities"]) for m in models] == [
        ("google/gemma-4-31b-it:free", ["completion"]),
        ("nvidia/nemotron-3-embed-1b:free", ["embedding"]),
    ]
    assert models[0]["display_name"] == "Gemma 4 31B (free)" and models[0]["vendor"] == "Google"
    assert models[0]["details"] == {"context_length": 262144}


@pytest.mark.asyncio
async def test_all_lists_paid_models_too(monkeypatch):
    _mark_ready(models="all")
    _fake_api(monkeypatch, _catalog)
    assert "openai/gpt-5" in [m["name"] for m in await openrouter_client.list_models()]


@pytest.mark.asyncio
async def test_chat_stream_sends_sampling_fields_and_yields_text(monkeypatch):
    _mark_ready()
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        sse = 'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\ndata: [DONE]\n\n'
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    _fake_api(monkeypatch, handler)
    chunks = [c async for c in openrouter_client.chat_stream("m:free", [{"role": "user", "content": "hi"}], _PARAMS)]
    assert "".join(chunks) == "Hi"
    assert seen["body"]["top_k"] == 40 and seen["body"]["repetition_penalty"] == 1.1
    assert "max_tokens" not in seen["body"] and "seed" not in seen["body"]  # -1 means none


@pytest.mark.asyncio
async def test_an_error_inside_the_stream_is_reported_not_swallowed(monkeypatch):
    _mark_ready()
    sse = (
        'data: {"choices":[{"delta":{"content":"Par"}}]}\n\n'
        'data: {"error":{"code":429,"message":"Rate limit exceeded"},"choices":[{"finish_reason":"error"}]}\n\n'
    )
    _fake_api(monkeypatch, lambda _r: httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"}))
    with pytest.raises(OpenRouterError, match="stopped the reply: Rate limit exceeded"):
        _ = [c async for c in openrouter_client.chat_stream("m:free", [{"role": "user", "content": "hi"}], _PARAMS)]


@pytest.mark.asyncio
async def test_embed_returns_the_vector(monkeypatch):
    _mark_ready()
    _fake_api(monkeypatch, lambda _r: httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}]}))
    assert await OpenRouterEngine().embed("hello", "nvidia/nemotron-3-embed-1b:free") == [0.1, 0.2]


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "ok", "fragment"), [(200, True, "Connected"), (401, False, "rejected")])
async def test_test_connection(monkeypatch, status, ok, fragment):
    _fake_api(monkeypatch, lambda _r: httpx.Response(status, json={"data": {}}))
    passed, message = await openrouter_client.test_connection({"api_key": "k"})
    assert passed is ok and fragment in message


@pytest.mark.asyncio
async def test_account_details_show_free_requests_and_credits(monkeypatch):
    _mark_ready()
    key = {
        "data": {
            "is_free_tier": True,
            "usage_daily": 0,
            "usage_monthly": 0.0123,
            "free_model_daily_requests": {"used": 3, "limit": 50, "remaining": 47},
        }
    }
    _fake_api(monkeypatch, lambda _r: httpx.Response(200, json=key))
    rows = dict(await OpenRouterEngine().account_details())
    assert rows["Key status"] == "Active" and rows["Listing"] == "Free models only"
    assert rows["Free requests today"] == "3 of 50"
    assert rows["Credits used this month"] == "$0.0123" and rows["Account"] == "Never bought credits"


@pytest.mark.asyncio
async def test_calls_refuse_until_ready():
    with pytest.raises(OpenRouterError, match="Connectors page"):
        await openrouter_client.list_models()
