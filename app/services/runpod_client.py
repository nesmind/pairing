"""Thin async wrapper around a RunPod Serverless endpoint's OpenAI-compatible API (the vLLM worker template;
https://docs.runpod.io/serverless/vllm/openai-compatibility).

Knows nothing of FastAPI or the UI, like ollama_client/matricxon_client. It reads its live config (endpoint
ID, API key, model) from connector_config_cache on every call, never bound at construction: an admin can
reconfigure or enable/disable it any time on the Connectors page (routers/connectors.py), as every engine adapter does.

RunPod's OpenAI-compatible surface has no embeddings route: RunPodEngine declares embeddings=False and never calls it.
"""

import logging
from collections.abc import AsyncGenerator

import httpx

from app.services import connector_config_cache
from app.services.openai_compat import OpenAIChatStream

logger = logging.getLogger("llama_chat")

_BASE_URL = "https://api.runpod.ai/v2"
_CONNECTOR_ID = "runpod"
# For a quick, non-generation call (list models, health).
_REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_HEALTH_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
# chat_stream's own, unbounded as ollama_client._CHAT_TIMEOUT: a fixed timeout would undercut
# reply_generation_service's.
_CHAT_TIMEOUT = httpx.Timeout(None, connect=10.0)


class RunPodError(Exception):
    """Raised when the RunPod endpoint is unreachable or errors, or RunPod isn't configured and enabled on the
    Connectors page, so routers/inference_client give a clean error rather than a stack trace or silent no-op.
    """


def _live_config() -> dict[str, str]:
    if not connector_config_cache.is_ready(_CONNECTOR_ID):
        raise RunPodError("RunPod Serverless isn't configured and enabled yet — set it up on the Connectors page.")
    return connector_config_cache.get_config(_CONNECTOR_ID)


def _openai_base(config: dict[str, str]) -> str:
    return f"{_BASE_URL}/{config['endpoint_id']}/openai/v1"


def _headers(config: dict[str, str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {config['api_key']}"}


async def list_models() -> list[dict]:
    """The model(s) the endpoint serves. Falls back to the admin's configured `model` if the live /models call
    fails (some vLLM deployments restrict that route).

    Every entry gets a "completion" capability though RunPod's /models doesn't report one: installed_chat_models/
    ChatModelCatalogBuilder.build (model_catalog_service) and the chat model switcher filter on it, so a working
    endpoint would otherwise show no models. Always correct here, since RunPodEngine's capabilities.embeddings is False.
    """
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
    """`params` is a GenerationParams-shaped dict (schemas/common.py). temperature/top_p map directly; num_predict
    becomes
    max_tokens (-1, our "unbounded", is omitted). top_k/repeat_penalty aren't OpenAI spec but vLLM's server accepts
    them as extra body fields, so they pass through.
    """
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
    """Streams a chat completion from RunPod's OpenAI-compatible endpoint as text chunks. `messages` is oldest-first
    {"role", "content"} dicts, the shape Ollama's chat_stream takes, so history forwards unchanged.
    """
    config = _live_config()
    payload = _chat_payload(model, messages, params, stream=True)
    url = f"{_openai_base(config)}/chat/completions"
    async for chunk in OpenAIChatStream("RunPod", RunPodError).stream(url, payload, _headers(config), _CHAT_TIMEOUT):
        yield chunk


async def chat_once(model: str, messages: list[dict], params: dict | None = None) -> str:
    """Non-streaming helper for one-off internal calls (e.g. titles); built on chat_stream like
    ollama_client.chat_once."""
    params = params or {"temperature": 0.3, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1, "num_predict": 64}
    text = ""
    async for chunk in chat_stream(model, messages, params):
        text += chunk
    return text


async def test_connection(config: dict[str, str]) -> tuple[bool, str]:
    """Validates a candidate config by hitting RunPod's /health with it; called from the Connectors page's "Test
    connection" (routers/connectors.py) against the saved config *before* it may make the connector ready
    (connector_config_cache.is_ready). Takes the config explicitly, unlike the rest of this module: an untested
    config isn't ready, so gating on is_ready (as _live_config does) would make the first test impossible.
    """
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
    """RunPod's /health: reachable or not (no host pool, as a Serverless endpoint has no hosts this app manages).
    Returns False without a network call when unconfigured.
    """
    if not connector_config_cache.is_ready(_CONNECTOR_ID):
        return {"runpod": False}
    config = connector_config_cache.get_config(_CONNECTOR_ID)
    async with httpx.AsyncClient(timeout=_HEALTH_TIMEOUT) as client:
        try:
            resp = await client.get(f"{_BASE_URL}/{config['endpoint_id']}/health", headers=_headers(config))
            return {"runpod": resp.status_code == 200}
        except httpx.HTTPError:
            return {"runpod": False}
