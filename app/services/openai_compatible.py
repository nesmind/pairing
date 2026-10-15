"""Shared client for cloud providers that speak the OpenAI API (`/models`, `/chat/completions`, `/embeddings`).
A provider is a subclass that sets its base URL and names and overrides only what differs: which models it lists,
which sampling fields it accepts, how it explains its own errors. Its key comes from connector_config_cache on every
call, like the other connector clients.
"""

from collections.abc import AsyncGenerator

import httpx

from app.services import connector_config_cache
from app.services.model_names import ReadableModelName
from app.services.openai_compat import OpenAIChatStream

REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
HEALTH_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
CHAT_TIMEOUT = httpx.Timeout(None, connect=10.0)  # unbounded, as the other connector clients


class OpenAICompatibleProvider:
    connector_id = ""
    name = ""  # as users know it, e.g. "NVIDIA NIM"
    base_url = ""
    key_help = ""  # where to create a key, for a rejected key
    error_type: type[Exception] = Exception

    # Connector config

    def live_config(self) -> dict[str, str]:
        if not connector_config_cache.is_ready(self.connector_id):
            raise self.error_type(f"{self.name} isn't set up yet — enable it on the Connectors page.")
        return connector_config_cache.get_config(self.connector_id)

    def headers(self, config: dict[str, str]) -> dict[str, str]:
        return {"Authorization": f"Bearer {config.get('api_key', '').strip()}"}

    # Models

    def model_info(self, entry: dict) -> dict | None:
        """A `/models` entry in the catalog's shape, or None to leave it out. Default: every model is a chat model."""
        model_id = entry.get("id", "")
        if not model_id:
            return None
        readable = ReadableModelName(model_id, fallback_vendor=self.name)
        return {
            "name": model_id,
            "capabilities": ["completion"],
            "display_name": readable.display_name,
            "vendor": readable.vendor,
            "details": {"context_length": entry.get("context_length") or entry.get("context_window")},
        }

    models_url = ""  # empty: `{base_url}/models`

    def model_entries(self, body: dict) -> list[dict]:
        """The model entries in the list endpoint's answer."""
        return body.get("data", [])

    async def _get_models(self, config: dict[str, str], timeout: httpx.Timeout) -> httpx.Response:
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.get(self.models_url or f"{self.base_url}/models", headers=self.headers(config))

    async def list_models(self) -> list[dict]:
        try:
            resp = await self._get_models(self.live_config(), REQUEST_TIMEOUT)
            resp.raise_for_status()
            entries = self.model_entries(resp.json())
        except httpx.HTTPError as exc:
            raise self.error_type(f"Could not list {self.name}'s models: {exc}") from exc
        models = [info for info in map(self.model_info, entries) if info is not None]
        return sorted(models, key=lambda m: m["name"])

    # Chat

    def chat_payload(self, model: str, messages: list[dict], params: dict, *, stream: bool) -> dict:
        """Temperature, Top-P and seed; text only. No reply-length cap by default: reasoning models think first, and
        a short cap such as a chat title's would leave an empty reply."""
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

    def explain(self, exc: Exception, model: str) -> Exception | None:
        """A clearer error for one of this provider's own failures, or None to keep `exc`."""
        return None

    async def chat_stream(self, model: str, messages: list[dict], params: dict) -> AsyncGenerator[str, None]:
        config = self.live_config()
        payload = self.chat_payload(model, messages, params, stream=True)
        try:
            async for chunk in OpenAIChatStream(self.name, self.error_type).stream(
                f"{self.base_url}/chat/completions", payload, self.headers(config), CHAT_TIMEOUT
            ):
                yield chunk
        except self.error_type as exc:
            clearer = self.explain(exc, model)
            if clearer is None:
                raise
            raise clearer from exc

    async def chat_once(self, model: str, messages: list[dict], params: dict | None = None) -> str:
        """Non-streaming helper for one-off internal calls (e.g. chat titles), built on chat_stream."""
        params = params or {"temperature": 0.3, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1, "num_predict": 64}
        return "".join([chunk async for chunk in self.chat_stream(model, messages, params)])

    # Embeddings

    def embed_payload(self, text: str, model: str) -> dict:
        return {"model": model, "input": text}

    async def embed(self, text: str, model: str) -> list[float]:
        return await self.post_embedding(self.embed_payload(text, model))

    async def post_embedding(self, payload: dict) -> list[float]:
        config = self.live_config()
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            try:
                resp = await client.post(f"{self.base_url}/embeddings", json=payload, headers=self.headers(config))
                resp.raise_for_status()
                return resp.json()["data"][0]["embedding"]
            except httpx.HTTPError as exc:
                raise self.error_type(f"{self.name} embedding request failed: {exc}") from exc
            except (KeyError, IndexError, ValueError) as exc:
                raise self.error_type(f"{self.name} returned an unexpected embedding response.") from exc

    # Health

    async def test_connection(self, config: dict[str, str]) -> tuple[bool, str]:
        """Checks a candidate config (explicitly given: an untested config isn't ready yet)."""
        if not config.get("api_key", "").strip():
            return False, "An API key is required."
        try:
            resp = await self._get_models(config, HEALTH_TIMEOUT)
        except httpx.HTTPError as exc:
            return False, f"Could not reach {self.name}: {exc}"
        if resp.status_code in (401, 403):
            return False, f"{self.name} rejected the API key — {self.key_help}"
        if resp.status_code != 200:
            return False, f"{self.name} returned HTTP {resp.status_code}."
        return True, "Connected successfully."

    async def check_health(self) -> dict[str, bool]:
        """Reachable or not; False without a network call when unconfigured."""
        if not connector_config_cache.is_ready(self.connector_id):
            return {self.connector_id: False}
        try:
            resp = await self._get_models(connector_config_cache.get_config(self.connector_id), HEALTH_TIMEOUT)
            return {self.connector_id: resp.status_code == 200}
        except httpx.HTTPError:
            return {self.connector_id: False}
