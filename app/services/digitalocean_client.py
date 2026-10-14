"""DigitalOcean Serverless Inference through its OpenAI-compatible API at https://inference.do-ai.run
(https://docs.digitalocean.com/products/inference/reference/api/serverless-inference/). Chat, embeddings and the model
list with one model access key; like runpod_client, the key comes from connector_config_cache on every call.
"""

import re
from collections.abc import AsyncGenerator

import httpx

from app.services import connector_config_cache
from app.services.model_names import ReadableModelName
from app.services.openai_compat import OpenAIChatStream

_BASE = "https://inference.do-ai.run/v1"
_CONNECTOR_ID = "digitalocean"
_REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_HEALTH_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
_CHAT_TIMEOUT = httpx.Timeout(None, connect=10.0)  # unbounded, as runpod_client
# /v1/models doesn't say a model's kind; embedding models are named for it.
_EMBEDDING_NAME = re.compile(r"embed|bge|gte|(^|[-_/])e5([-_]|$)|mini-?lm|mpnet|nomic", re.IGNORECASE)
# Listed too, but not text models (image, video, speech, rerankers): left out so no chat picks one.
_NOT_TEXT_NAME = re.compile(r"image|stable-diffusion|flux|t2v|i2v|tts|whisper|diarization|reranker", re.IGNORECASE)


class DigitalOceanError(Exception):
    """DigitalOcean is unreachable, refused the call, or isn't configured and enabled on the Connectors page."""


def _live_config() -> dict[str, str]:
    if not connector_config_cache.is_ready(_CONNECTOR_ID):
        raise DigitalOceanError("DigitalOcean isn't set up yet — enable it on the Connectors page.")
    return connector_config_cache.get_config(_CONNECTOR_ID)


def _headers(config: dict[str, str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {config.get('api_key', '').strip()}"}


def _model_info(model_id: str) -> dict:
    """One model in the shape the catalog reads (model_catalog_service._auto_discovered_entry)."""
    readable = ReadableModelName(model_id, fallback_vendor="DigitalOcean")
    return {
        "name": model_id,
        "capabilities": ["embedding" if _EMBEDDING_NAME.search(model_id) else "completion"],
        "display_name": readable.display_name,
        "vendor": readable.vendor,
        "details": {},
    }


async def get_models(config: dict[str, str]) -> httpx.Response:
    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
        return await client.get(f"{_BASE}/models", headers=_headers(config))


async def list_models() -> list[dict]:
    try:
        resp = await get_models(_live_config())
        resp.raise_for_status()
        ids = [entry["id"] for entry in resp.json().get("data", []) if entry.get("id")]
    except httpx.HTTPError as exc:
        raise DigitalOceanError(f"Could not list DigitalOcean's models: {exc}") from exc
    return sorted((_model_info(i) for i in ids if not _NOT_TEXT_NAME.search(i)), key=lambda m: m["name"])


def _chat_payload(model: str, messages: list[dict], params: dict, *, stream: bool) -> dict:
    """Only the fields DigitalOcean documents for /v1/chat/completions (no top_k or repeat penalty there);
    text only, so other message keys (e.g. images) are dropped."""
    payload = {
        "model": model,
        "messages": [{"role": m["role"], "content": m.get("content") or ""} for m in messages],
        "stream": stream,
        "temperature": params["temperature"],
        "top_p": params["top_p"],
    }
    if params.get("num_predict", -1) != -1:
        payload["max_tokens"] = params["num_predict"]
    if params.get("seed", -1) not in (-1, None):
        payload["seed"] = params["seed"]
    return payload


async def chat_stream(model: str, messages: list[dict], params: dict) -> AsyncGenerator[str, None]:
    config = _live_config()
    payload = _chat_payload(model, messages, params, stream=True)
    async for chunk in OpenAIChatStream("DigitalOcean", DigitalOceanError).stream(
        f"{_BASE}/chat/completions", payload, _headers(config), _CHAT_TIMEOUT
    ):
        yield chunk


async def chat_once(model: str, messages: list[dict], params: dict | None = None) -> str:
    """Non-streaming helper for one-off internal calls (e.g. chat titles), built on chat_stream."""
    params = params or {"temperature": 0.3, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1, "num_predict": 64}
    return "".join([chunk async for chunk in chat_stream(model, messages, params)])


async def embed(text: str, model: str) -> list[float]:
    config = _live_config()
    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
        try:
            resp = await client.post(
                f"{_BASE}/embeddings", json={"model": model, "input": text}, headers=_headers(config)
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]
        except httpx.HTTPError as exc:
            raise DigitalOceanError(f"DigitalOcean embedding request failed: {exc}") from exc
        except (KeyError, IndexError, ValueError) as exc:
            raise DigitalOceanError("DigitalOcean returned an unexpected embedding response.") from exc


async def test_connection(config: dict[str, str]) -> tuple[bool, str]:
    """Checks a candidate config (explicitly given: an untested config isn't ready yet, see runpod_client)."""
    if not config.get("api_key", "").strip():
        return False, "A model access key is required."
    try:
        resp = await get_models(config)
    except httpx.HTTPError as exc:
        return False, f"Could not reach DigitalOcean: {exc}"
    if resp.status_code in (401, 403):
        return False, "DigitalOcean rejected the key — use a model access key from Serverless Inference."
    if resp.status_code != 200:
        return False, f"DigitalOcean returned HTTP {resp.status_code}."
    return True, "Connected successfully."


async def check_health() -> dict[str, bool]:
    """Reachable or not; False without a network call when unconfigured."""
    if not connector_config_cache.is_ready(_CONNECTOR_ID):
        return {"digitalocean": False}
    try:
        resp = await get_models(connector_config_cache.get_config(_CONNECTOR_ID))
        return {"digitalocean": resp.status_code == 200}
    except httpx.HTTPError:
        return {"digitalocean": False}


async def rate_limit_details() -> list[tuple[str, str]]:
    """The hourly rate limit DigitalOcean reports in response headers, for the External servers summary."""
    try:
        resp = await get_models(connector_config_cache.get_config(_CONNECTOR_ID))
    except httpx.HTTPError:
        return []
    remaining, limit = resp.headers.get("ratelimit-remaining"), resp.headers.get("ratelimit-limit")
    rows = [("Key status", "Active" if resp.status_code == 200 else f"HTTP {resp.status_code}")]
    if remaining and limit:
        rows.append(("Requests left", f"{remaining} of {limit}"))
    return rows
