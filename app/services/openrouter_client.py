"""OpenRouter (https://openrouter.ai/docs): one key for hundreds of models from many providers, through an
OpenAI-compatible API. By default only the free (":free") models are listed, so testing costs nothing. Like the other
connector clients, it reads its config from connector_config_cache on every call.
"""

from collections.abc import AsyncGenerator

import httpx

from app.services import connector_config_cache
from app.services.openai_compat import OpenAIChatStream

_BASE = "https://openrouter.ai/api/v1"
_CONNECTOR_ID = "openrouter"
_REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_HEALTH_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
_CHAT_TIMEOUT = httpx.Timeout(None, connect=10.0)  # unbounded, as the other connector clients
# OpenRouter's optional app attribution headers.
_APP_HEADERS = {"HTTP-Referer": "https://upupabit.ai", "X-Title": "pAIring"}


class OpenRouterError(Exception):
    """OpenRouter is unreachable, refused the call, or isn't configured and enabled on the Connectors page."""


def _live_config() -> dict[str, str]:
    if not connector_config_cache.is_ready(_CONNECTOR_ID):
        raise OpenRouterError("OpenRouter isn't set up yet — enable it on the Connectors page.")
    return connector_config_cache.get_config(_CONNECTOR_ID)


def _headers(config: dict[str, str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {config.get('api_key', '').strip()}", **_APP_HEADERS}


def free_only(config: dict[str, str]) -> bool:
    """The "Models to list" field: anything but "all" lists only the free models."""
    return config.get("models", "").strip().lower() != "all"


def _is_free(entry: dict) -> bool:
    """Every listed price is zero (-1 marks a router with variable pricing, so not free)."""
    pricing = entry.get("pricing") or {}
    try:
        return bool(pricing) and all(float(value) == 0 for value in pricing.values())
    except (TypeError, ValueError):
        return False


def _text_only(entry: dict) -> bool:
    """Leaves out models that also make audio or images (e.g. music models): chat shows text only."""
    outputs = (entry.get("architecture") or {}).get("output_modalities") or ["text"]
    return set(outputs) == {"text"}


def _model_info(entry: dict, capability: str) -> dict:
    """OpenRouter names read "Vendor: Model (free)"; the catalog wants them apart."""
    vendor, _, name = (entry.get("name") or entry["id"]).partition(": ")
    if not name:
        vendor, name = entry["id"].split("/", 1)[0].replace("-", " ").title(), vendor
    return {
        "name": entry["id"],
        "capabilities": [capability],
        "display_name": name.replace("  ", " ").strip(),
        "vendor": "OpenRouter" if vendor.strip().lower() == "openrouter" else vendor.strip(),
        "details": {"context_length": entry.get("context_length")},
    }


async def list_models() -> list[dict]:
    """Chat models ("completion") and embedding models ("embedding"); free ones only unless set to "all"."""
    config = _live_config()
    models: list[dict] = []
    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
        try:
            for path, capability in (("/models", "completion"), ("/embeddings/models", "embedding")):
                resp = await client.get(f"{_BASE}{path}", headers=_headers(config))
                resp.raise_for_status()
                for entry in resp.json().get("data", []):
                    wanted = not free_only(config) or _is_free(entry)
                    if entry.get("id") and wanted and (capability == "embedding" or _text_only(entry)):
                        models.append(_model_info(entry, capability))
        except httpx.HTTPError as exc:
            raise OpenRouterError(f"Could not list OpenRouter's models: {exc}") from exc
    return sorted(models, key=lambda m: m["name"])


def _chat_payload(model: str, messages: list[dict], params: dict, *, stream: bool) -> dict:
    """OpenRouter accepts the sampling fields below and ignores those a model lacks; text only for now."""
    payload = {
        "model": model,
        "messages": [{"role": m["role"], "content": m.get("content") or ""} for m in messages],
        "stream": stream,
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "top_k": params["top_k"],
        "repetition_penalty": params["repeat_penalty"],
    }
    if params.get("num_predict", -1) != -1:
        payload["max_tokens"] = params["num_predict"]
    if params.get("seed", -1) not in (-1, None):
        payload["seed"] = params["seed"]
    return payload


async def chat_stream(model: str, messages: list[dict], params: dict) -> AsyncGenerator[str, None]:
    config = _live_config()
    payload = _chat_payload(model, messages, params, stream=True)
    async for chunk in OpenAIChatStream("OpenRouter", OpenRouterError).stream(
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
            raise OpenRouterError(f"OpenRouter embedding request failed: {exc}") from exc
        except (KeyError, IndexError, ValueError) as exc:
            raise OpenRouterError("OpenRouter returned an unexpected embedding response.") from exc


async def key_info(config: dict[str, str]) -> httpx.Response:
    """GET /api/v1/key: the key's limits, credit use and free-model requests today."""
    async with httpx.AsyncClient(timeout=_HEALTH_TIMEOUT) as client:
        return await client.get(f"{_BASE}/key", headers=_headers(config))


async def test_connection(config: dict[str, str]) -> tuple[bool, str]:
    """Checks a candidate config (explicitly given: an untested config isn't ready yet)."""
    if not config.get("api_key", "").strip():
        return False, "An API key is required."
    try:
        resp = await key_info(config)
    except httpx.HTTPError as exc:
        return False, f"Could not reach OpenRouter: {exc}"
    if resp.status_code in (401, 403):
        return False, "OpenRouter rejected the API key — create one under Settings > Keys on openrouter.ai."
    if resp.status_code != 200:
        return False, f"OpenRouter returned HTTP {resp.status_code}."
    return True, "Connected successfully."


async def check_health() -> dict[str, bool]:
    """Reachable or not; False without a network call when unconfigured."""
    if not connector_config_cache.is_ready(_CONNECTOR_ID):
        return {"openrouter": False}
    try:
        return {"openrouter": (await key_info(connector_config_cache.get_config(_CONNECTOR_ID))).status_code == 200}
    except httpx.HTTPError:
        return {"openrouter": False}


async def account_details() -> list[tuple[str, str]]:
    """Key status, free-model requests today and credit use, for the External servers summary."""
    config = connector_config_cache.get_config(_CONNECTOR_ID)
    try:
        resp = await key_info(config)
        body = resp.json()
        data = body.get("data", body) or {}
    except (httpx.HTTPError, ValueError):
        return [("Key status", "could not check")]
    if resp.status_code != 200:
        return [("Key status", f"HTTP {resp.status_code}")]
    rows = [("Key status", "Active"), ("Listing", "Free models only" if free_only(config) else "All models")]
    free = data.get("free_model_daily_requests") or {}
    if free.get("limit") is not None:
        rows.append(("Free requests today", f"{free.get('used', 0)} of {free['limit']}"))
    rows.append(("Credits used today", f"${float(data.get('usage_daily') or 0):.4f}"))
    rows.append(("Credits used this month", f"${float(data.get('usage_monthly') or 0):.4f}"))
    rows.append(("Account", "Never bought credits" if data.get("is_free_tier") else "Has bought credits"))
    return rows
