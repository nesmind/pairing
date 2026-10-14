"""Tests for services/cloudflare_client.py (Cloudflare Workers AI through its OpenAI-compatible API) and the
shared stream reader in services/openai_compat.py. httpx.AsyncClient gets a MockTransport: no network."""

import json

import httpx
import pytest

from app.services import cloudflare_client, connector_config_cache
from app.services.cloudflare_client import CloudflareError, CloudflareModelName
from app.services.engines.cloudflare_engine import CloudflareEngine
from app.services.openai_compat import OpenAIChatStream

_CONFIG = {"account_id": "acc-1", "api_token": "tok"}
_PARAMS = {"temperature": 0.5, "top_p": 0.9, "top_k": 120, "repeat_penalty": 1.1, "num_predict": 64, "seed": 7}
_BASE = "https://api.cloudflare.com/client/v4/accounts/acc-1/ai"


@pytest.fixture(autouse=True)
def reset_cache():
    connector_config_cache._cache = {}
    yield
    connector_config_cache._cache = {}


def _mark_ready():
    connector_config_cache._cache = {
        "cloudflare": {"config": _CONFIG, "enabled": True, "configured": True, "last_test_passed": True}
    }


def _fake_api(monkeypatch, handler):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))


@pytest.mark.asyncio
async def test_calls_refuse_until_the_connector_is_ready():
    with pytest.raises(CloudflareError, match="Connectors page"):
        await cloudflare_client.list_models()
    assert await cloudflare_client.check_health() == {"cloudflare": False}


@pytest.mark.asyncio
async def test_list_models_tags_chat_and_embedding_models_and_skips_non_chat(monkeypatch):
    _mark_ready()
    by_task = {
        "Text Generation": [
            "@cf/meta/llama-3.2-1b-instruct",
            "@cf/google/gemma-7b-it-lora",
            "@cf/meta/llama-guard-3-8b",
        ],
        "Text Embeddings": ["@cf/baai/bge-small-en-v1.5"],
    }

    def handler(request):
        assert request.url.path == "/client/v4/accounts/acc-1/ai/models/search"
        assert request.headers["Authorization"] == "Bearer tok"
        names = by_task[request.url.params["task"]]
        ctx = [{"property_id": "context_window", "value": "60000"}]
        return httpx.Response(200, json={"success": True, "result": [{"name": n, "properties": ctx} for n in names]})

    _fake_api(monkeypatch, handler)
    models = await cloudflare_client.list_models()
    assert [(m["name"], m["capabilities"]) for m in models] == [
        ("@cf/baai/bge-small-en-v1.5", ["embedding"]),
        ("@cf/meta/llama-3.2-1b-instruct", ["completion"]),
    ]
    assert models[1]["display_name"] == "Llama 3.2 1B Instruct" and models[1]["vendor"] == "Meta"
    assert models[1]["details"] == {"context_length": 60000}


@pytest.mark.asyncio
async def test_chat_stream_sends_workers_ai_fields_and_yields_text(monkeypatch):
    _mark_ready()
    seen = {}

    def handler(request):
        seen["url"], seen["body"] = str(request.url), json.loads(request.content)
        sse = 'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\ndata: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        return httpx.Response(200, text=sse + "data: [DONE]\n\n", headers={"content-type": "text/event-stream"})

    _fake_api(monkeypatch, handler)
    messages = [{"role": "user", "content": "hi", "images": ["abc"]}]
    chunks = [c async for c in cloudflare_client.chat_stream("@cf/m", messages, _PARAMS)]

    assert "".join(chunks) == "Hello"
    assert seen["url"] == f"{_BASE}/v1/chat/completions"
    body = seen["body"]
    assert body["messages"] == [{"role": "user", "content": "hi"}]  # text only
    assert body["top_k"] == 50 and body["repetition_penalty"] == 1.1  # clamped / renamed
    assert body["max_tokens"] == 64 and body["seed"] == 7 and body["stream"] is True


@pytest.mark.asyncio
async def test_chat_stream_reports_cloudflares_error_text(monkeypatch):
    _mark_ready()
    _fake_api(monkeypatch, lambda _r: httpx.Response(400, json={"errors": [{"message": "No such model"}]}))
    with pytest.raises(CloudflareError, match="400.*No such model"):
        _ = [c async for c in cloudflare_client.chat_stream("@cf/x", [{"role": "user", "content": "hi"}], _PARAMS)]


@pytest.mark.asyncio
async def test_embed_returns_the_vector(monkeypatch):
    _mark_ready()

    def handler(request):
        assert request.url.path.endswith("/ai/v1/embeddings")
        assert json.loads(request.content) == {"model": "@cf/baai/bge-small-en-v1.5", "input": "hello"}
        return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2, 0.3]}]})

    _fake_api(monkeypatch, handler)
    assert await CloudflareEngine().embed("hello", "@cf/baai/bge-small-en-v1.5") == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected_ok", "fragment"),
    [
        (200, True, "Connected"),
        (400, False, "rejected the API token"),
        (403, False, "Workers AI permission"),
        (404, False, "account ID looks wrong"),
    ],
)
async def test_test_connection_explains_the_result(monkeypatch, status, expected_ok, fragment):
    _fake_api(monkeypatch, lambda _r: httpx.Response(status, json={}))
    ok, message = await cloudflare_client.test_connection(_CONFIG)
    assert ok is expected_ok and fragment in message


@pytest.mark.asyncio
async def test_test_connection_needs_both_fields():
    ok, message = await cloudflare_client.test_connection({"account_id": "acc-1", "api_token": " "})
    assert ok is False and "required" in message


def test_stream_reader_handles_done_and_empty_lines():
    assert OpenAIChatStream.content_of('data: {"choices":[{"delta":{"content":"x"}}]}') == "x"
    assert OpenAIChatStream.content_of('data: {"choices":[{"delta":{}}]}') == ""
    assert OpenAIChatStream.content_of("") == ""
    assert OpenAIChatStream.content_of("data: [DONE]") is None


def test_the_engine_declares_embeddings_and_no_local_process():
    capabilities = CloudflareEngine.capabilities
    assert capabilities.embeddings is True
    assert capabilities.local_process is False and capabilities.model_management is False


@pytest.mark.parametrize(
    ("model_id", "name", "vendor"),
    [
        ("@cf/openai/gpt-oss-120b", "GPT OSS 120B", "OpenAI"),
        ("@cf/meta/llama-3.1-8b-instruct-fp8", "Llama 3.1 8B Instruct FP8", "Meta"),
        ("@cf/deepseek-ai/deepseek-r1-distill-qwen-32b", "DeepSeek R1 Distill Qwen 32B", "DeepSeek"),
        ("@cf/some-lab/new-model-7b", "New Model 7B", "Some Lab"),
    ],
)
def test_model_names_read_well(model_id, name, vendor):
    readable = CloudflareModelName(model_id)
    assert readable.display_name == name and readable.vendor == vendor
