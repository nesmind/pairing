"""Groq (https://console.groq.com/docs/openai): fast hosted open models through an OpenAI-compatible API, with a free
tier that needs no card. Chat only: Groq has no embeddings. Reads its key from connector_config_cache on every call.
"""

import re
from collections.abc import AsyncGenerator

import httpx

from app.services import connector_config_cache
from app.services.model_names import ReadableModelName
from app.services.openai_compat import OpenAIChatStream

_BASE = "https://api.groq.com/openai/v1"
_CONNECTOR_ID = "groq"
_REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_HEALTH_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
_CHAT_TIMEOUT = httpx.Timeout(None, connect=10.0)  # unbounded, as the other connector clients
# Listed with the chat models but not text chat: speech-to-text, text-to-speech and safety classifiers.
_NOT_TEXT_NAME = re.compile(r"whisper|tts|orpheus|playai|guard|safeguard|distil", re.IGNORECASE)
_VENDORS = {"meta": "Meta", "openai": "OpenAI", "alibaba cloud": "Qwen", "moonshot ai": "Moonshot AI"}


class GroqError(Exception):
    """Groq is unreachable, refused the call, or isn't configured and enabled on the Connectors page."""


def _live_config() -> dict[str, str]:
    if not connector_config_cache.is_ready(_CONNECTOR_ID):
        raise GroqError("Groq isn't set up yet — enable it on the Connectors page.")
    return connector_config_cache.get_config(_CONNECTOR_ID)


def _headers(config: dict[str, str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {config.get('api_key', '').strip()}"}


async def _models(config: dict[str, str], timeout: httpx.Timeout = _REQUEST_TIMEOUT) -> httpx.Response:
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await client.get(f"{_BASE}/models", headers=_headers(config))


def _model_info(entry: dict) -> dict | None:
    """An active text chat model in the catalog's shape, else None."""
    model_id = entry.get("id", "")
    if not model_id or entry.get("active") is False or _NOT_TEXT_NAME.search(model_id):
        return None
    readable = ReadableModelName(model_id, fallback_vendor=entry.get("owned_by") or "Groq")
    owner = (entry.get("owned_by") or "").strip()
    return {
        "name": model_id,
        "capabilities": ["completion"],
        "display_name": readable.display_name,
        "vendor": _VENDORS.get(owner.lower(), owner) or readable.vendor,
        "details": {"context_length": entry.get("context_window")},
    }


async def list_models() -> list[dict]:
    try:
        resp = await _models(_live_config())
        resp.raise_for_status()
        entries = resp.json().get("data", [])
    except httpx.HTTPError as exc:
        raise GroqError(f"Could not list Groq's models: {exc}") from exc
    models = [info for info in map(_model_info, entries) if info is not None]
    return sorted(models, key=lambda m: m["name"])


def _chat_payload(model: str, messages: list[dict], params: dict, *, stream: bool) -> dict:
    """Temperature, Top-P and seed (Groq has no top_k or repeat penalty); text only. No reply-length cap: reasoning
    models think first, and a short cap such as a chat title's would leave an empty reply."""
    payload = {
        "model": model,
        "messages": [{"role": m["role"], "content": m.get("content") or ""} for m in messages],
        "stream": stream,
        "temperature": params["temperature"],
        "top_p": params["top_p"],
    }
    if params.get("seed", -1) not in (-1, None):
        payload["seed"] = params["seed"]
    return payload


async def chat_stream(model: str, messages: list[dict], params: dict) -> AsyncGenerator[str, None]:
    config = _live_config()
    payload = _chat_payload(model, messages, params, stream=True)
    async for chunk in OpenAIChatStream("Groq", GroqError).stream(
        f"{_BASE}/chat/completions", payload, _headers(config), _CHAT_TIMEOUT
    ):
        yield chunk


async def chat_once(model: str, messages: list[dict], params: dict | None = None) -> str:
    """Non-streaming helper for one-off internal calls (e.g. chat titles), built on chat_stream."""
    params = params or {"temperature": 0.3, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1, "num_predict": 64}
    return "".join([chunk async for chunk in chat_stream(model, messages, params)])


async def test_connection(config: dict[str, str]) -> tuple[bool, str]:
    """Checks a candidate config (explicitly given: an untested config isn't ready yet)."""
    if not config.get("api_key", "").strip():
        return False, "An API key is required."
    try:
        resp = await _models(config, _HEALTH_TIMEOUT)
    except httpx.HTTPError as exc:
        return False, f"Could not reach Groq: {exc}"
    if resp.status_code in (401, 403):
        return False, "Groq rejected the API key — create one at console.groq.com under API Keys."
    if resp.status_code != 200:
        return False, f"Groq returned HTTP {resp.status_code}."
    return True, "Connected successfully."


async def check_health() -> dict[str, bool]:
    """Reachable or not; False without a network call when unconfigured."""
    if not connector_config_cache.is_ready(_CONNECTOR_ID):
        return {"groq": False}
    try:
        resp = await _models(connector_config_cache.get_config(_CONNECTOR_ID), _HEALTH_TIMEOUT)
        return {"groq": resp.status_code == 200}
    except httpx.HTTPError:
        return {"groq": False}
