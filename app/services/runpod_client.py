"""
Thin async wrapper around a RunPod Serverless endpoint's OpenAI-compatible API (the
vLLM worker template — see https://docs.runpod.io/serverless/vllm/openai-compatibility).

Nothing here knows about FastAPI or the web UI — it only knows how to talk to RunPod,
the same boundary app.services.ollama_client/matricxon_client keep for their own
engines. Reads its own live config (endpoint ID, API key, model) from
app.services.connector_config_cache on every call — never bound at construction,
since an admin can (re)configure or enable/disable this at any time from the
Connectors page (see app/routers/connectors.py), and every existing engine adapter
already follows this same "look up fresh, every call" rule.

RunPod's OpenAI-compatible surface has no embeddings route — app.services.engines.
runpod_engine.RunPodEngine declares embeddings=False and never calls anything here
for it.
"""

import json
import logging
from collections.abc import AsyncGenerator

import httpx

from app.services import connector_config_cache

logger = logging.getLogger("llama_chat")

_BASE_URL = "https://api.runpod.ai/v2"
_CONNECTOR_ID = "runpod"
# For a quick, non-generation call (list models, health).
_REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_HEALTH_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
# chat_stream's own — unbounded, matching ollama_client._CHAT_TIMEOUT's own reasoning:
# a fixed timeout here would silently undercut reply_generation_service's own.
_CHAT_TIMEOUT = httpx.Timeout(None, connect=10.0)


class RunPodError(Exception):
    """Raised when the configured RunPod endpoint is unreachable or returns an
    error, or when RunPod Serverless hasn't been configured and enabled yet on the
    Connectors page — so routers/inference_client turn this into a clean error
    instead of a raw stack trace or a silent no-op."""


def _live_config() -> dict[str, str]:
    if not connector_config_cache.is_ready(_CONNECTOR_ID):
        raise RunPodError("RunPod Serverless isn't configured and enabled yet — set it up on the Connectors page.")
    return connector_config_cache.get_config(_CONNECTOR_ID)


def _openai_base(config: dict[str, str]) -> str:
    return f"{_BASE_URL}/{config['endpoint_id']}/openai/v1"


def _headers(config: dict[str, str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {config['api_key']}"}


async def list_models() -> list[dict]:
    """The model(s) the configured endpoint currently serves. Falls back to the
    admin's own configured `model` field if the live /models call fails — some vLLM
    deployments restrict that route.

    Every entry gets a real "completion" capability tag even though RunPod's own
    OpenAI-compatible /models route doesn't report one — app.services.model_catalog_
    service's own installed_chat_models()/ChatModelCatalogBuilder.build() (and the
    chat page's own model-switch picker, which calls the former) filter on
    "completion" in capabilities before a model is even considered installed at all;
    without this, a real, fully working RunPod endpoint would show as having no
    installed models anywhere in the app. "completion" is always correct here since
    RunPodEngine's own capabilities.embeddings is False - this app never asks a
    RunPod-served model for anything else."""
    config = _live_config()
    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
        try:
            resp = await client.get(f"{_openai_base(config)}/models", headers=_headers(config))
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if data:
                return [{"name": entry["id"], "capabilities": ["completion"]} for entry in data]
        except httpx.HTTPError as exc:
            logger.warning("RunPod GET /models failed, falling back to the configured model name: %s", exc)
    model = config.get("model")
    if not model:
        raise RunPodError("Could not list RunPod's models and no fallback model name is configured.")
    return [{"name": model, "capabilities": ["completion"]}]


def _chat_payload(model: str, messages: list[dict], params: dict, *, stream: bool) -> dict:
    """`params` is a GenerationParams-shaped dict (see app/schemas/common.py).
    temperature/top_p map directly to OpenAI's own fields; num_predict becomes
    max_tokens (-1, our "unbounded" sentinel, is simply omitted). top_k/repeat_penalty
    aren't part of the OpenAI spec, but vLLM's own OpenAI-compatible server accepts
    them as plain extra body fields, so they're passed straight through."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "top_k": params["top_k"],
        "repeat_penalty": params["repeat_penalty"],
    }
    if params.get("num_predict", -1) != -1:
        payload["max_tokens"] = params["num_predict"]
    return payload


async def chat_stream(model: str, messages: list[dict], params: dict) -> AsyncGenerator[str, None]:
    """Streams a chat completion from RunPod's OpenAI-compatible endpoint, yielding
    text chunks as they arrive. `messages` is a list of {"role": ..., "content": ...}
    dicts, oldest first — the same OpenAI-compatible shape Ollama's own chat_stream
    already expects, so callers forward conversation history unchanged."""
    config = _live_config()
    payload = _chat_payload(model, messages, params, stream=True)
    async with httpx.AsyncClient(timeout=_CHAT_TIMEOUT) as client:
        try:
            async with client.stream(
                "POST", f"{_openai_base(config)}/chat/completions", json=payload, headers=_headers(config)
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    choices = json.loads(data).get("choices") or [{}]
                    content = choices[0].get("delta", {}).get("content", "")
                    if content:
                        yield content
        except httpx.HTTPError as exc:
            raise RunPodError(f"RunPod chat request failed: {exc}") from exc


async def chat_once(model: str, messages: list[dict], params: dict | None = None) -> str:
    """Non-streaming helper for one-off internal calls (e.g. conversation-title
    generation) where only the final text is needed. Built on chat_stream, same as
    app.services.ollama_client.chat_once."""
    params = params or {"temperature": 0.3, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1, "num_predict": 64}
    text = ""
    async for chunk in chat_stream(model, messages, params):
        text += chunk
    return text


async def test_connection(config: dict[str, str]) -> tuple[bool, str]:
    """Validates a candidate config by actually hitting RunPod's own /health endpoint with it — called from the
    Connectors page's "Test connection" action (app/routers/connectors.py) against whatever is currently saved,
    *before* that config is allowed to make this connector ready (see app.services.connector_config_cache.is_ready).
    Takes the config explicitly rather than reading connector_config_cache, unlike every other function in this
    module: a config isn't ready yet at the moment it's being tested, so gating this on is_ready() the way
    _live_config() does would make it impossible to ever pass the first test."""
    endpoint_id = config.get("endpoint_id")
    api_key = config.get("api_key")
    if not endpoint_id or not api_key:
        return False, "Endpoint ID and API key are both required."
    async with httpx.AsyncClient(timeout=_HEALTH_TIMEOUT) as client:
        try:
            resp = await client.get(f"{_BASE_URL}/{endpoint_id}/health", headers={"Authorization": f"Bearer {api_key}"})
        except httpx.HTTPError as exc:
            return False, f"Could not reach RunPod: {exc}"
    if resp.status_code == 401:
        return False, "RunPod rejected the API key (401 Unauthorized)."
    if resp.status_code == 404:
        return False, "Endpoint ID not found (404) — double-check it."
    if resp.status_code != 200:
        return False, f"RunPod returned HTTP {resp.status_code}."
    return True, "Connected successfully."


async def check_health() -> dict[str, bool]:
    """RunPod's own /health endpoint — reachable-or-not, not a per-host pool the way
    Ollama/Matricxon's check_health is, since a Serverless endpoint has no concept of
    "hosts" this app manages. Returns False without a network call when unconfigured,
    matching the rest of this module's "not ready" handling."""
    if not connector_config_cache.is_ready(_CONNECTOR_ID):
        return {"runpod": False}
    config = connector_config_cache.get_config(_CONNECTOR_ID)
    async with httpx.AsyncClient(timeout=_HEALTH_TIMEOUT) as client:
        try:
            resp = await client.get(f"{_BASE_URL}/{config['endpoint_id']}/health", headers=_headers(config))
            return {"runpod": resp.status_code == 200}
        except httpx.HTTPError:
            return {"runpod": False}
