"""Google Gemini API (https://ai.google.dev/gemini-api/docs/openai): chat and embeddings through its OpenAI-compatible
endpoint, and the model list from the native API, which says what each model is for (generateContent vs embedContent)
and gives its name and context size. One API key; read from connector_config_cache on every call.
"""

import re
from collections.abc import AsyncGenerator

import httpx

from app.services import connector_config_cache
from app.services.openai_compat import OpenAIChatStream

_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
_NATIVE_MODELS = "https://generativelanguage.googleapis.com/v1beta/models"
_CONNECTOR_ID = "gemini"
_REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_HEALTH_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
_CHAT_TIMEOUT = httpx.Timeout(None, connect=10.0)  # unbounded, as the other connector clients
# Listed with generateContent but not text chat: images, speech, transcription, live audio, video, music, robotics,
# and long-running agents (Deep Research, Antigravity).
_NOT_TEXT_NAME = re.compile(
    r"image|banana|tts|transcribe|live|audio|veo|imagen|lyria|robotics|computer-use|native|deep-research|antigravity",
    re.IGNORECASE,
)


class GeminiError(Exception):
    """Gemini is unreachable, refused the call, or isn't configured and enabled on the Connectors page."""


def _live_config() -> dict[str, str]:
    if not connector_config_cache.is_ready(_CONNECTOR_ID):
        raise GeminiError("Gemini isn't set up yet — enable it on the Connectors page.")
    return connector_config_cache.get_config(_CONNECTOR_ID)


def _key(config: dict[str, str]) -> str:
    return config.get("api_key", "").strip()


async def _native_models(config: dict[str, str], timeout: httpx.Timeout = _REQUEST_TIMEOUT) -> httpx.Response:
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await client.get(_NATIVE_MODELS, params={"pageSize": 1000}, headers={"x-goog-api-key": _key(config)})


def _model_info(entry: dict) -> dict | None:
    """A native model entry in the catalog's shape, or None when it isn't a text chat or embedding model."""
    model_id = entry.get("name", "").removeprefix("models/")
    methods = entry.get("supportedGenerationMethods") or []
    if "embedContent" in methods:
        capability = "embedding"
    elif "generateContent" in methods and not _NOT_TEXT_NAME.search(model_id):
        capability = "completion"
    else:
        return None
    return {
        "name": model_id,
        "capabilities": [capability],
        "display_name": entry.get("displayName") or model_id,
        "vendor": "Google",
        "details": {"context_length": entry.get("inputTokenLimit")},
    }


async def list_models() -> list[dict]:
    try:
        resp = await _native_models(_live_config())
        resp.raise_for_status()
        entries = resp.json().get("models", [])
    except httpx.HTTPError as exc:
        raise GeminiError(f"Could not list Gemini's models: {exc}") from exc
    models = [info for info in map(_model_info, entries) if info is not None]
    return sorted(models, key=lambda m: m["name"])


def _chat_payload(model: str, messages: list[dict], params: dict, *, stream: bool) -> dict:
    """Temperature and Top-P only; text only. Not sent, as tested live: `seed` (rejected as an unknown field) and a
    reply-length cap (Gemini models think first and the thinking counts toward it, so a short cap such as the 64
    tokens of a chat title left an empty reply). No top_k or repeat penalty in Google's OpenAI-compatible API."""
    return {
        "model": model,
        "messages": [{"role": m["role"], "content": m.get("content") or ""} for m in messages],
        "stream": stream,
        "temperature": params["temperature"],
        "top_p": params["top_p"],
    }


def _headers(config: dict[str, str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {_key(config)}"}


async def chat_stream(model: str, messages: list[dict], params: dict) -> AsyncGenerator[str, None]:
    config = _live_config()
    payload = _chat_payload(model, messages, params, stream=True)
    try:
        async for chunk in OpenAIChatStream("Gemini", GeminiError).stream(
            f"{_OPENAI_BASE}/chat/completions", payload, _headers(config), _CHAT_TIMEOUT
        ):
            yield chunk
    except GeminiError as exc:
        if "(500)" in str(exc):  # Google's own internal error, seen for every request to some models (e.g. Gemma)
            raise GeminiError(
                f"Google returned an internal error for {model}. This happens on Google's side for some models; "
                "try another model, such as a Gemini Flash one."
            ) from exc
        raise


async def chat_once(model: str, messages: list[dict], params: dict | None = None) -> str:
    """Non-streaming helper for one-off internal calls (e.g. chat titles), built on chat_stream."""
    params = params or {"temperature": 0.3, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1, "num_predict": 64}
    return "".join([chunk async for chunk in chat_stream(model, messages, params)])


async def embed(text: str, model: str) -> list[float]:
    config = _live_config()
    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
        try:
            resp = await client.post(
                f"{_OPENAI_BASE}/embeddings", json={"model": model, "input": text}, headers=_headers(config)
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]
        except httpx.HTTPError as exc:
            raise GeminiError(f"Gemini embedding request failed: {exc}") from exc
        except (KeyError, IndexError, ValueError) as exc:
            raise GeminiError("Gemini returned an unexpected embedding response.") from exc


async def test_connection(config: dict[str, str]) -> tuple[bool, str]:
    """Checks a candidate config (explicitly given: an untested config isn't ready yet)."""
    if not _key(config):
        return False, "An API key is required."
    try:
        resp = await _native_models(config, _HEALTH_TIMEOUT)
    except httpx.HTTPError as exc:
        return False, f"Could not reach Gemini: {exc}"
    if resp.status_code in (400, 401, 403):
        return False, "Google rejected the API key — create one in Google AI Studio (Get API key)."
    if resp.status_code != 200:
        return False, f"Gemini returned HTTP {resp.status_code}."
    return True, "Connected successfully."


async def check_health() -> dict[str, bool]:
    """Reachable or not; False without a network call when unconfigured."""
    if not connector_config_cache.is_ready(_CONNECTOR_ID):
        return {"gemini": False}
    try:
        resp = await _native_models(connector_config_cache.get_config(_CONNECTOR_ID), _HEALTH_TIMEOUT)
        return {"gemini": resp.status_code == 200}
    except httpx.HTTPError:
        return {"gemini": False}
