"""Tests for services/gemini_client.py (Google Gemini: native model list, OpenAI-compatible chat and embeddings).
httpx.AsyncClient gets a MockTransport: no network."""

import json

import httpx
import pytest

from app.services import connector_config_cache, gemini_client
from app.services.engines.gemini_engine import GeminiEngine
from app.services.gemini_client import GeminiError

_PARAMS = {"temperature": 0.5, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1, "num_predict": 64, "seed": 7}
_NATIVE = [
    {
        "name": "models/gemini-3.8-flash",
        "displayName": "Gemini 3.8 Flash",
        "inputTokenLimit": 1048576,
        "supportedGenerationMethods": ["generateContent", "countTokens"],
    },
    {
        "name": "models/gemma-4-31b-it",
        "displayName": "Gemma 4 31B",
        "inputTokenLimit": 131072,
        "supportedGenerationMethods": ["generateContent"],
    },
    {
        "name": "models/gemini-embedding-001",
        "displayName": "Gemini Embedding 001",
        "inputTokenLimit": 2048,
        "supportedGenerationMethods": ["embedContent"],
    },
    {
        "name": "models/gemini-3.8-flash-image",
        "displayName": "Nano Banana",
        "supportedGenerationMethods": ["generateContent"],
    },
    {"name": "models/gemini-3.8-flash-tts", "displayName": "TTS", "supportedGenerationMethods": ["generateContent"]},
    {"name": "models/aqa", "displayName": "AQA", "supportedGenerationMethods": ["generateAnswer"]},
    {"name": "models/deep-research-preview-04-2026", "supportedGenerationMethods": ["generateContent"]},
    {"name": "models/gemini-3.5-transcribe", "supportedGenerationMethods": ["generateContent"]},
    {"name": "models/nano-banana-pro-preview", "supportedGenerationMethods": ["generateContent"]},
]


@pytest.fixture(autouse=True)
def reset_cache():
    connector_config_cache._cache = {}
    yield
    connector_config_cache._cache = {}


def _mark_ready():
    connector_config_cache._cache = {
        "gemini": {"config": {"api_key": "AIza-key"}, "enabled": True, "configured": True, "last_test_passed": True}
    }


def _fake_api(monkeypatch, handler):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))


@pytest.mark.asyncio
async def test_lists_text_chat_and_embedding_models_with_their_names(monkeypatch):
    _mark_ready()

    def handler(request):
        assert request.url.path == "/v1beta/models" and request.headers["x-goog-api-key"] == "AIza-key"
        return httpx.Response(200, json={"models": _NATIVE})

    _fake_api(monkeypatch, handler)
    models = {m["name"]: m for m in await gemini_client.list_models()}

    assert set(models) == {"gemini-3.8-flash", "gemma-4-31b-it", "gemini-embedding-001"}  # no image/tts/aqa
    assert models["gemini-3.8-flash"]["capabilities"] == ["completion"]
    assert models["gemini-embedding-001"]["capabilities"] == ["embedding"]
    assert (
        models["gemini-3.8-flash"]["display_name"] == "Gemini 3.8 Flash"
        and models["gemini-3.8-flash"]["vendor"] == "Google"
    )
    assert models["gemini-3.8-flash"]["details"] == {"context_length": 1048576}


@pytest.mark.asyncio
async def test_chat_uses_the_openai_endpoint_with_documented_fields(monkeypatch):
    _mark_ready()
    seen = {}

    def handler(request):
        seen["url"], seen["auth"], seen["body"] = (
            str(request.url),
            request.headers["Authorization"],
            json.loads(request.content),
        )
        sse = 'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\ndata: [DONE]\n\n'
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    _fake_api(monkeypatch, handler)
    chunks = [
        c async for c in gemini_client.chat_stream("gemini-3.8-flash", [{"role": "user", "content": "hi"}], _PARAMS)
    ]

    assert "".join(chunks) == "Hello"
    assert seen["url"] == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    assert seen["auth"] == "Bearer AIza-key"
    # No seed (rejected) and no reply-length cap (thinking would use it up): see _chat_payload.
    assert set(seen["body"]) == {"model", "messages", "stream", "temperature", "top_p"}


@pytest.mark.asyncio
async def test_embed_returns_the_vector(monkeypatch):
    _mark_ready()

    def handler(request):
        assert request.url.path == "/v1beta/openai/embeddings"
        return httpx.Response(200, json={"data": [{"embedding": [0.3, 0.4]}]})

    _fake_api(monkeypatch, handler)
    assert await GeminiEngine().embed("hello", "gemini-embedding-001") == [0.3, 0.4]


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "ok", "fragment"), [(200, True, "Connected"), (400, False, "rejected")])
async def test_test_connection(monkeypatch, status, ok, fragment):
    _fake_api(monkeypatch, lambda _r: httpx.Response(status, json={"models": []}))
    passed, message = await gemini_client.test_connection({"api_key": "k"})
    assert passed is ok and fragment in message


@pytest.mark.asyncio
async def test_usage_points_to_ai_studio_and_mentions_free_tier_use():
    usage, note = await GeminiEngine().usage_summary()
    assert usage is None and "AI Studio" in note and "improve" in note


@pytest.mark.asyncio
async def test_calls_refuse_until_ready():
    with pytest.raises(GeminiError, match="Connectors page"):
        await gemini_client.list_models()


@pytest.mark.asyncio
async def test_googles_internal_error_gets_a_clear_message(monkeypatch):
    _mark_ready()
    body = [{"error": {"code": 500, "message": "Internal error encountered.", "status": "INTERNAL"}}]
    _fake_api(monkeypatch, lambda _r: httpx.Response(500, json=body))
    with pytest.raises(GeminiError, match="internal error for gemma-4-31b-it.*try another model"):
        _ = [c async for c in gemini_client.chat_stream("gemma-4-31b-it", [{"role": "user", "content": "hi"}], _PARAMS)]
