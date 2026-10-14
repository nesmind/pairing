"""Tests for services/digitalocean_client.py (DigitalOcean Serverless Inference through its OpenAI-compatible API)
and services/model_names.py. httpx.AsyncClient gets a MockTransport: no network."""

import json

import httpx
import pytest

from app.services import connector_config_cache, digitalocean_client
from app.services.digitalocean_client import DigitalOceanError
from app.services.engines.digitalocean_engine import DigitalOceanEngine
from app.services.model_names import ReadableModelName

_CONFIG = {"api_key": "sk-do-key"}
_PARAMS = {"temperature": 0.5, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1, "num_predict": 64, "seed": 7}


@pytest.fixture(autouse=True)
def reset_cache():
    connector_config_cache._cache = {}
    yield
    connector_config_cache._cache = {}


def _mark_ready():
    connector_config_cache._cache = {
        "digitalocean": {"config": _CONFIG, "enabled": True, "configured": True, "last_test_passed": True}
    }


def _fake_api(monkeypatch, handler):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))


@pytest.mark.asyncio
async def test_calls_refuse_until_the_connector_is_ready():
    with pytest.raises(DigitalOceanError, match="Connectors page"):
        await digitalocean_client.list_models()
    assert await digitalocean_client.check_health() == {"digitalocean": False}


@pytest.mark.asyncio
async def test_list_models_names_them_and_tells_embedding_models_apart(monkeypatch):
    _mark_ready()

    def handler(request):
        assert str(request.url) == "https://inference.do-ai.run/v1/models"
        assert request.headers["Authorization"] == "Bearer sk-do-key"
        ids = [
            "anthropic-claude-3.5-sonnet",
            "gte-large-en-v1.5",
            "all-mini-lm-l6-v2",
            "llama3.3-70b-instruct",
            "openai-gpt-image-1",
            "wan2-2-t2v-a14b",
            "bge-reranker-v2-m3",
        ]
        return httpx.Response(200, json={"object": "list", "data": [{"id": i, "object": "model"} for i in ids]})

    _fake_api(monkeypatch, handler)
    models = {m["name"]: m for m in await digitalocean_client.list_models()}

    assert models["anthropic-claude-3.5-sonnet"]["capabilities"] == ["completion"]
    assert models["anthropic-claude-3.5-sonnet"]["display_name"] == "Claude 3.5 Sonnet"
    assert models["anthropic-claude-3.5-sonnet"]["vendor"] == "Anthropic"
    assert models["gte-large-en-v1.5"]["capabilities"] == ["embedding"]
    assert models["all-mini-lm-l6-v2"]["capabilities"] == ["embedding"]
    assert not {"openai-gpt-image-1", "wan2-2-t2v-a14b", "bge-reranker-v2-m3"} & set(models)  # not text models
    assert models["llama3.3-70b-instruct"]["vendor"] == "Meta"


@pytest.mark.asyncio
async def test_chat_stream_sends_only_documented_fields(monkeypatch):
    _mark_ready()
    seen = {}

    def handler(request):
        seen["url"], seen["body"] = str(request.url), json.loads(request.content)
        sse = 'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\ndata: [DONE]\n\n'
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    _fake_api(monkeypatch, handler)
    messages = [{"role": "user", "content": "hello", "images": ["abc"]}]
    chunks = [c async for c in digitalocean_client.chat_stream("llama3.3-70b-instruct", messages, _PARAMS)]

    assert "".join(chunks) == "Hi"
    assert seen["url"] == "https://inference.do-ai.run/v1/chat/completions"
    assert set(seen["body"]) == {"model", "messages", "stream", "temperature", "top_p", "max_tokens", "seed"}
    assert seen["body"]["messages"] == [{"role": "user", "content": "hello"}]


@pytest.mark.asyncio
async def test_embed_returns_the_vector(monkeypatch):
    _mark_ready()

    def handler(request):
        assert request.url.path == "/v1/embeddings"
        return httpx.Response(200, json={"object": "list", "data": [{"embedding": [0.5, 0.25], "index": 0}]})

    _fake_api(monkeypatch, handler)
    assert await DigitalOceanEngine().embed("hello", "gte-large-en-v1.5") == [0.5, 0.25]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected_ok", "fragment"),
    [(200, True, "Connected"), (401, False, "model access key"), (500, False, "HTTP 500")],
)
async def test_test_connection_explains_the_result(monkeypatch, status, expected_ok, fragment):
    _fake_api(monkeypatch, lambda _r: httpx.Response(status, json={"data": []}))
    ok, message = await digitalocean_client.test_connection(_CONFIG)
    assert ok is expected_ok and fragment in message


@pytest.mark.asyncio
async def test_rate_limit_comes_from_the_response_headers(monkeypatch):
    _mark_ready()
    headers = {"ratelimit-remaining": "4990", "ratelimit-limit": "5000"}
    _fake_api(monkeypatch, lambda _r: httpx.Response(200, json={"data": []}, headers=headers))
    rows = await DigitalOceanEngine().account_details()
    assert ("Key status", "Active") in rows and ("Requests left", "4990 of 5000") in rows


@pytest.mark.asyncio
async def test_usage_points_to_the_control_panel():
    usage, note = await DigitalOceanEngine().usage_summary()
    assert usage is None and "control panel" in note


@pytest.mark.parametrize(
    ("model_id", "name", "vendor"),
    [
        ("openai-gpt-4o", "GPT 4o", "OpenAI"),
        ("alibaba-qwen3-32b", "Qwen3 32B", "Qwen"),
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", "Meta Llama 3.1 8B Instruct", "Meta"),
        ("mistral-nemo-instruct-2407", "Mistral Nemo Instruct 2407", "Mistral AI"),
        ("unknown-thing", "Unknown Thing", "DigitalOcean"),
    ],
)
def test_readable_names(model_id, name, vendor):
    readable = ReadableModelName(model_id, fallback_vendor="DigitalOcean")
    assert readable.display_name == name and readable.vendor == vendor
