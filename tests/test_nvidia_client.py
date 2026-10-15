"""Tests for NVIDIA NIM (nvidia_client, on openai_compatible.OpenAICompatibleProvider).
httpx.AsyncClient gets a MockTransport: no network."""

import json

import httpx
import pytest

from app.services import connector_config_cache, nvidia_client
from app.services.engines.nvidia_engine import NvidiaEngine
from app.services.nvidia_client import NvidiaError

PARAMS = {"temperature": 0.5, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1, "num_predict": 64, "seed": 7}
SSE = 'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\ndata: [DONE]\n\n'


@pytest.fixture(autouse=True)
def ready():
    connector_config_cache._cache = {
        cid: {"config": {"api_key": "k"}, "enabled": True, "configured": True, "last_test_passed": True}
        for cid in ("nvidia",)
    }
    yield
    connector_config_cache._cache = {}


def _fake_api(monkeypatch, handler):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))


def _chat_body(monkeypatch, provider, params) -> dict:
    seen = {}

    def handler(request):
        seen["url"], seen["body"] = str(request.url), json.loads(request.content)
        return httpx.Response(200, text=SSE, headers={"content-type": "text/event-stream"})

    _fake_api(monkeypatch, handler)
    import asyncio

    assert asyncio.run(provider.chat_once("m", [{"role": "user", "content": "hi"}], params)) == "Hi"
    return seen


class TestNvidia:
    @pytest.mark.asyncio
    async def test_lists_text_chat_and_embedding_models_only(self, monkeypatch):
        ids = [
            "meta/llama-3.3-70b-instruct",
            "nvidia/llama-3.2-nv-embedqa-1b-v2",
            "nvidia/llama-3.1-nemoguard-8b-content-safety",
            "nvidia/nv-rerankqa-mistral-4b-v3",
            "microsoft/phi-3.5-vision-instruct",
            "nvidia/parakeet-ctc-1.1b-asr",
            "deepseek-ai/deepseek-r1",
        ]
        _fake_api(monkeypatch, lambda r: httpx.Response(200, json={"data": [{"id": i} for i in ids]}))
        models = {m["name"]: m["capabilities"] for m in await nvidia_client.PROVIDER.list_models()}
        assert models == {
            "meta/llama-3.3-70b-instruct": ["completion"],
            "nvidia/llama-3.2-nv-embedqa-1b-v2": ["embedding"],
            "deepseek-ai/deepseek-r1": ["completion"],
        }

    @pytest.mark.parametrize(("num_predict", "max_tokens"), [(64, 1024), (-1, 4096), (3000, 3000)])
    def test_chat_always_sends_a_reply_length(self, monkeypatch, num_predict, max_tokens):
        seen = _chat_body(monkeypatch, nvidia_client.PROVIDER, {**PARAMS, "num_predict": num_predict})
        assert seen["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
        assert seen["body"]["max_tokens"] == max_tokens and seen["body"]["seed"] == 7

    @pytest.mark.asyncio
    async def test_embeds_with_query_mode_then_without_it_when_refused(self, monkeypatch):
        bodies = []

        def handler(request):
            body = json.loads(request.content)
            bodies.append(body)
            if "input_type" in body and body["model"] == "baai/bge-m3":
                return httpx.Response(400, json={"detail": "input_type not supported"})
            return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}]})

        _fake_api(monkeypatch, handler)
        assert await NvidiaEngine().embed("hello", "nvidia/nv-embedqa-e5-v5") == [0.1, 0.2]
        assert bodies[-1]["input_type"] == "query"
        assert await NvidiaEngine().embed("hello", "baai/bge-m3") == [0.1, 0.2]
        assert "input_type" not in bodies[-1]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("status", "fragment"), [(402, "free credits"), (429, "free credits"), (404, "no hosted")])
    async def test_credit_rate_and_missing_model_errors_are_explained(self, monkeypatch, status, fragment):
        _fake_api(monkeypatch, lambda r: httpx.Response(status, json={"detail": "x"}))
        with pytest.raises(NvidiaError, match=fragment):
            await nvidia_client.PROVIDER.chat_once("m", [{"role": "user", "content": "hi"}], PARAMS)


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "ok", "fragment"), [(200, True, "Connected"), (401, False, "rejected")])
async def test_test_connection(monkeypatch, status, ok, fragment):
    _fake_api(monkeypatch, lambda r: httpx.Response(status, json={"data": []}))
    passed, message = await nvidia_client.PROVIDER.test_connection({"api_key": "k"})
    assert passed is ok and fragment in message


@pytest.mark.asyncio
async def test_calls_refuse_until_ready_and_usage_points_to_build_nvidia_com():
    connector_config_cache._cache = {}
    with pytest.raises(NvidiaError, match="Connectors page"):
        await nvidia_client.PROVIDER.list_models()
    assert "build.nvidia.com" in (await NvidiaEngine().usage_summary())[1]
