"""Cloudflare Workers AI through its OpenAI-compatible API (https://developers.cloudflare.com/workers-ai/).
Chat, embeddings and the model list; no local process. Like runpod_client, it reads its live config (account
ID, API token) from connector_config_cache on every call, so the Connectors page can change it any time.
"""

import logging
import re
from collections.abc import AsyncGenerator

import httpx

from app.services import connector_config_cache
from app.services.openai_compat import OpenAIChatStream

logger = logging.getLogger("llama_chat")

_API = "https://api.cloudflare.com/client/v4/accounts"
_CONNECTOR_ID = "cloudflare"
_REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_HEALTH_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
_CHAT_TIMEOUT = httpx.Timeout(None, connect=10.0)  # unbounded, as runpod_client
# Model-search task -> the capability pAIring's pickers filter on (model_catalog_service).
_TASKS = {"Text Generation": "completion", "Text Embeddings": "embedding"}
# Not plain chat models: LoRA bases need an adapter, Llama Guard only classifies safety.
_SKIP_MARKERS = ("-lora", "llama-guard")
# Workers AI limits for the extra sampling fields.
_MAX_TOP_K = 50
# Model-id org -> how its name reads; others are title-cased.
_VENDORS = {
    "meta": "Meta",
    "meta-llama": "Meta",
    "google": "Google",
    "qwen": "Qwen",
    "mistral": "Mistral AI",
    "mistralai": "Mistral AI",
    "deepseek-ai": "DeepSeek",
    "openai": "OpenAI",
    "baai": "BAAI",
    "moonshotai": "Moonshot AI",
    "zai-org": "Z.ai",
    "ibm-granite": "IBM",
    "nvidia": "NVIDIA",
    "aisingapore": "AI Singapore",
}
_UPPER_WORDS = {"gpt", "oss", "glm", "bge", "fp8", "it", "r1", "en", "vl", "hf"}
_WORD_FIXES = {"deepseek": "DeepSeek"}
_SIZE_WORD = re.compile(r"[a-z]?\d+(\.\d+)?[bm]", re.IGNORECASE)  # 8b, 0.6b, a12b


class CloudflareModelName:
    """Readable name and vendor from an id like "@cf/meta/llama-3.2-1b-instruct"."""

    def __init__(self, model_id: str) -> None:
        parts = model_id.split("/")
        self._org = parts[1] if len(parts) > 2 else ""
        self._model = parts[-1]

    @property
    def vendor(self) -> str:
        return _VENDORS.get(self._org, self._org.replace("-", " ").title() or "Cloudflare")

    @property
    def display_name(self) -> str:
        words = []
        for word in self._model.split("-"):
            if word.lower() in _WORD_FIXES:
                words.append(_WORD_FIXES[word.lower()])
            elif _SIZE_WORD.fullmatch(word) or word.lower() in _UPPER_WORDS:
                words.append(word.upper())
            else:
                words.append(word[:1].upper() + word[1:])
        return " ".join(words)


class CloudflareError(Exception):
    """Cloudflare is unreachable, refused the call, or isn't configured and enabled on the Connectors page."""


def _live_config() -> dict[str, str]:
    if not connector_config_cache.is_ready(_CONNECTOR_ID):
        raise CloudflareError("Cloudflare Workers AI isn't set up yet — enable it on the Connectors page.")
    return connector_config_cache.get_config(_CONNECTOR_ID)


def _base(config: dict[str, str]) -> str:
    return f"{_API}/{config['account_id'].strip()}/ai"


def _headers(config: dict[str, str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {config['api_token'].strip()}"}


async def _search(client: httpx.AsyncClient, config: dict[str, str], task: str) -> list[dict]:
    resp = await client.get(
        f"{_base(config)}/models/search", params={"task": task, "per_page": 100}, headers=_headers(config)
    )
    resp.raise_for_status()
    return resp.json().get("result", [])


async def list_models() -> list[dict]:
    """Chat models tagged "completion" and embedding models tagged "embedding"."""
    config = _live_config()
    models: list[dict] = []
    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
        try:
            for task, capability in _TASKS.items():
                for entry in await _search(client, config, task):
                    name = entry.get("name", "")
                    if name and not any(marker in name for marker in _SKIP_MARKERS):
                        models.append(_model_info(name, capability, entry))
        except httpx.HTTPError as exc:
            raise CloudflareError(f"Could not list Cloudflare's models: {exc}") from exc
    return sorted(models, key=lambda m: m["name"])


def _model_info(name: str, capability: str, entry: dict) -> dict:
    """One model in the shape the catalog reads (model_catalog_service._auto_discovered_entry)."""
    props = {p.get("property_id"): p.get("value") for p in entry.get("properties", [])}
    try:
        context_length = int(props.get("context_window") or 0) or None
    except (TypeError, ValueError):
        context_length = None
    readable = CloudflareModelName(name)
    return {
        "name": name,
        "capabilities": [capability],
        "display_name": readable.display_name,
        "vendor": readable.vendor,
        "details": {"context_length": context_length},
    }


def _chat_payload(model: str, messages: list[dict], params: dict, *, stream: bool) -> dict:
    """GenerationParams -> Workers AI fields; text only, so other message keys (e.g. images) are dropped."""
    payload = {
        "model": model,
        "messages": [{"role": m["role"], "content": m.get("content") or ""} for m in messages],
        "stream": stream,
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "top_k": max(1, min(params["top_k"], _MAX_TOP_K)),
        "repetition_penalty": params["repeat_penalty"],
    }
    if params.get("num_predict", -1) != -1:
        payload["max_tokens"] = params["num_predict"]
    if params.get("seed", -1) not in (-1, None):
        payload["seed"] = params["seed"]
    return payload


async def chat_stream(model: str, messages: list[dict], params: dict) -> AsyncGenerator[str, None]:
    config = _live_config()
    url = f"{_base(config)}/v1/chat/completions"
    payload = _chat_payload(model, messages, params, stream=True)
    async for chunk in OpenAIChatStream("Cloudflare", CloudflareError).stream(
        url, payload, _headers(config), _CHAT_TIMEOUT
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
                f"{_base(config)}/v1/embeddings", json={"model": model, "input": text}, headers=_headers(config)
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]
        except httpx.HTTPError as exc:
            raise CloudflareError(f"Cloudflare embedding request failed: {exc}") from exc
        except (KeyError, IndexError, ValueError) as exc:
            raise CloudflareError("Cloudflare returned an unexpected embedding response.") from exc


async def _probe(config: dict[str, str], timeout: httpx.Timeout) -> httpx.Response:
    """A cheap, free call: one model from the search."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await client.get(f"{_base(config)}/models/search", params={"per_page": 1}, headers=_headers(config))


async def test_connection(config: dict[str, str]) -> tuple[bool, str]:
    """Checks a candidate config (explicitly given: an untested config isn't ready yet, see runpod_client)."""
    if not config.get("account_id", "").strip() or not config.get("api_token", "").strip():
        return False, "Account ID and API token are both required."
    try:
        resp = await _probe(config, _HEALTH_TIMEOUT)
    except httpx.HTTPError as exc:
        return False, f"Could not reach Cloudflare: {exc}"
    if resp.status_code in (400, 401):
        return False, "Cloudflare rejected the API token — check that it was copied in full."
    if resp.status_code == 403:
        return False, "Access refused: check the account ID, and that the token has the Workers AI permission for it."
    if resp.status_code == 404:
        return False, "The account ID looks wrong — copy it from the Cloudflare dashboard."
    if resp.status_code != 200:
        return False, f"Cloudflare returned HTTP {resp.status_code}."
    return True, "Connected successfully."


async def check_health() -> dict[str, bool]:
    """Reachable or not; False without a network call when unconfigured."""
    if not connector_config_cache.is_ready(_CONNECTOR_ID):
        return {"cloudflare": False}
    try:
        resp = await _probe(connector_config_cache.get_config(_CONNECTOR_ID), _HEALTH_TIMEOUT)
        return {"cloudflare": resp.status_code == 200}
    except httpx.HTTPError:
        return {"cloudflare": False}
