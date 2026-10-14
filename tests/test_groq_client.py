"""Tests for services/groq_client.py (Groq: OpenAI-compatible model list and chat; no embeddings).
httpx.AsyncClient gets a MockTransport: no network."""

import json

import httpx
import pytest

from app.services import connector_config_cache, groq_client
from app.services.engines.base import EngineCapabilityError
from app.services.engines.groq_engine import GroqEngine
from app.services.groq_client import GroqError

_PARAMS = {"temperature": 0.5, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1, "num_predict": 64, "seed": 7}
_MODELS = [
    {"id": "llama-3.3-70b-versatile", "owned_by": "Meta", "active": True, "context_window": 131072},
    {"id": "openai/gpt-oss-120b", "owned_by": "OpenAI", "active": True, "context_window": 131072},
    {"id": "qwen/qwen3-32b", "owned_by": "Alibaba Cloud", "active": True, "context_window": 131072},
    {"id": "whisper-large-v3", "owned_by": "OpenAI", "active": True, "context_window": 448},
    {"id": "playai-tts", "owned_by": "PlayAI", "active": True, "context_window": 8192},
    {"id": "meta-llama/llama-guard-4-12b", "owned_by": "Meta", "active": True, "context_window": 131072},
    {"id": "old-model", "owned_by": "Meta", "active": False, "context_window": 8192},
]


@pytest.fixture(autouse=True)
def reset_cache():
    connector_config_cache._cache = {}
    yield
    connector_config_cache._cache = {}


def _mark_ready():
    connector_config_cache._cache = {
        "groq": {"config": {"api_key": "gsk-key"}, "enabled": True, "configured": True, "last_test_passed": True}
    }


def _fake_api(monkeypatch, handler):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))


@pytest.mark.asyncio
async def test_lists_active_text_chat_models_with_names(monkeypatch):
    _mark_ready()

    def handler(request):
        assert request.url.path == "/openai/v1/models" and request.headers["Authorization"] == "Bearer gsk-key"
        return httpx.Response(200, json={"data": _MODELS})

    _fake_api(monkeypatch, handler)
    models = {m["name"]: m for m in await groq_client.list_models()}

    assert set(models) == {"llama-3.3-70b-versatile", "openai/gpt-oss-120b", "qwen/qwen3-32b"}
    assert models["openai/gpt-oss-120b"]["display_name"] == "GPT OSS 120B"
    assert models["openai/gpt-oss-120b"]["vendor"] == "OpenAI"
    assert models["qwen/qwen3-32b"]["vendor"] == "Qwen"
    assert models["llama-3.3-70b-versatile"]["details"] == {"context_length": 131072}
    assert all(m["capabilities"] == ["completion"] for m in models.values())


@pytest.mark.asyncio
async def test_chat_sends_supported_fields_only(monkeypatch):
    _mark_ready()
    seen = {}

    def handler(request):
        seen["url"], seen["body"] = str(request.url), json.loads(request.content)
        sse = 'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\ndata: [DONE]\n\n'
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    _fake_api(monkeypatch, handler)
    reply = await groq_client.chat_once("llama-3.3-70b-versatile", [{"role": "user", "content": "hi"}], _PARAMS)

    assert reply == "Hi"
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert set(seen["body"]) == {"model", "messages", "stream", "temperature", "top_p", "seed"}


def test_seed_left_out_when_random():
    payload = groq_client._chat_payload("m", [], {**_PARAMS, "seed": -1}, stream=True)
    assert "seed" not in payload


@pytest.mark.asyncio
async def test_embeddings_are_refused():
    assert GroqEngine.capabilities.embeddings is False
    with pytest.raises(EngineCapabilityError):
        await GroqEngine().embed("hello", "any")


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "ok", "fragment"), [(200, True, "Connected"), (401, False, "rejected")])
async def test_test_connection(monkeypatch, status, ok, fragment):
    _fake_api(monkeypatch, lambda _r: httpx.Response(status, json={"data": []}))
    passed, message = await groq_client.test_connection({"api_key": "k"})
    assert passed is ok and fragment in message


@pytest.mark.asyncio
async def test_usage_points_to_the_console():
    usage, note = await GroqEngine().usage_summary()
    assert usage is None and "console.groq.com" in note


@pytest.mark.asyncio
async def test_calls_refuse_until_ready():
    with pytest.raises(GroqError, match="Connectors page"):
        await groq_client.list_models()
