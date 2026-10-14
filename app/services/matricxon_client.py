"""Thin async wrapper around Matricxon's local HTTP API, the analogue of ollama_client.py and deliberately
separate: the backends are unrelated processes configured independently (engine_service), and separate
plumbing keeps their failure/cooldown state and error types from leaking. Matricxon's API is Ollama-shaped
(../matricxon/app/main.py calls it "an Ollama-API-compatible inference server"), matched here against its
router/schema code (tags_router.py, chat_router.py, embeddings_router.py), not Ollama's docs.

Deliberate gaps versus ollama_client.py:
  - No "thinking" stripping (ollama_thinking): Matricxon implements only mistral3/bert/nomic-bert today
    (../matricxon/app/architectures/registry.py), none thinking-capable. So chat_stream also skips
    ollama_client.chat_stream's nested _content_stream, which only lets that check pick a generator.
  - Embeddings use Matricxon's singular `/api/embeddings` (not Ollama's `/api/embed`); same shapes otherwise
    (../matricxon/app/schemas/embeddings.py).

OpenTelemetry spans (attribute shaping: matricxon_telemetry, kept out of this file like ollama_telemetry)
mirror ollama_client.py's chat_stream/embed spans field for field: its done-line payload uses the same
prompt_eval_count/eval_count/load_duration/eval_duration/total_duration names (checked
against../matricxon/app/routers/chat_router.py).
"""

import json
import logging
from collections.abc import AsyncGenerator

import httpx
from opentelemetry.trace import SpanKind, Status, StatusCode

from app.config import EMBEDDING_NUM_CTX
from app.services import matricxon_http, matricxon_pool, matricxon_telemetry
from app.services.tool_call_sink import add_tools_to_payload, sink_from

logger = logging.getLogger("llama_chat")

_REQUEST_TIMEOUT = httpx.Timeout(120.0, connect=5.0)
_CHAT_TIMEOUT = httpx.Timeout(None, connect=5.0)


class MatricxonError(Exception):
    """Raised when Matricxon is unreachable or errors; like ollama_client.OllamaError, a separate type (no shared
    base) so one engine's errors can't be caught as the other's. inference_client normalizes both into InferenceError.
    """


async def list_models() -> list[dict]:
    """Models pulled into Matricxon (GET /api/tags): same name/capabilities/details shape as Ollama's
    (../matricxon/app/routers/tags_router.py), so model_catalog_service builds a catalog from either engine.

    Never cached, unlike get_capabilities: installed state must be fresh, or "just pulled/uninstalled, UI behind"
    returns.
    """
    host = matricxon_pool.pick_host()
    async with matricxon_http.client(timeout=_REQUEST_TIMEOUT) as client:
        try:
            resp = await client.get(f"{host}/api/tags")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise MatricxonError(f"Could not reach Matricxon at {host}: {exc}") from exc
    return resp.json().get("models", [])


async def get_capabilities() -> dict:
    """Matricxon's current support surface: GET /api/health (../matricxon/app/routers/health_router.py), a
    Matricxon-only extension, so callers ask "can you run this?" instead of keeping a driftable copy of its
    architecture/quantization registries (../matricxon/ROADMAP.md "Expose supported architectures/quantizations";
    only caller: MatricxonSupportChecker). Returns `{"supported_architectures": [...], "supported_quantizations":
    [...]}`:
    every `general.architecture` and GGML quant name it can load. It asks Matricxon specifically, not the active
    engine, so it never goes through inference_client's dispatch.

    Never cached. It was cached 30s on the theory that support changes only on upgrade+restart, which broke: Matricxon's
    registry gained granite/granitemoe/nemotron_h in one session without pAIring
    restarting, and the stale answer kept saying "not supported" with no way to refresh. /api/health is cheap
    (HealthRequestHandler.handle returns two in-memory lists), so the saved redundancy (three catalog builders
    asking per page load) isn't worth the lost accuracy.
    """
    host = matricxon_pool.pick_host()
    async with matricxon_http.client(timeout=_REQUEST_TIMEOUT) as client:
        try:
            resp = await client.get(f"{host}/api/health")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise MatricxonError(f"Could not reach Matricxon at {host}: {exc}") from exc
    return resp.json()


async def embed(text: str, model: str) -> list[float]:
    """Embeds `text` via POST /api/embeddings (singular) with `model` (the admin's default, resolved by callers via
    model_catalog_service.get_default_embedding_model). No cold-start retry (see ollama_client.embed): Matricxon's
    ModelManager loads the model inside the request (../matricxon/app/routers/embeddings_router.py), so there's no
    "dispatched before ready" race, and one span attempt_count always.
    """
    host = matricxon_pool.pick_host()
    with matricxon_telemetry.get_tracer().start_as_current_span(
        "matricxon.embed", kind=SpanKind.CLIENT, record_exception=False, set_status_on_exception=False
    ) as span:
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("server.address", host)
        span.set_attribute("ollama.attempt_count", 1)
        async with matricxon_pool.track_request(host):
            async with matricxon_http.client(timeout=_REQUEST_TIMEOUT) as client:
                try:
                    resp = await client.post(
                        f"{host}/api/embeddings",
                        json={"model": model, "prompt": text, "options": {"num_ctx": EMBEDDING_NUM_CTX}},
                    )
                    resp.raise_for_status()
                except httpx.HTTPError as exc:
                    matricxon_pool.mark_failure(host)
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise MatricxonError(f"Embedding request failed: {exc}") from exc
        matricxon_pool.mark_recovered(host)
    return resp.json()["embedding"]


def _error_detail_from_body(body: bytes, status_code: int) -> str:
    """Extracts Matricxon's `{"error": "..."}` detail from an error body (the shape of every typed error
    in../matricxon/app/server/errors.py). Falls back to the raw text (e.g. an unhandled exception's plain body, before
    its catch-all handler) and only to a generic message when there is nothing.
    """
    try:
        detail = json.loads(body).get("error")
    except (json.JSONDecodeError, AttributeError):
        detail = None
    if not detail:
        detail = body.decode("utf-8", errors="replace").strip() or None
    return detail or f"Matricxon returned HTTP {status_code} with no error detail."


async def chat_stream(model: str, messages: list[dict], params: dict) -> AsyncGenerator[str, None]:
    """Streams a chat completion as text chunks; see ollama_client.chat_stream for the shared contract (`messages`
    role/content dicts in). Matches../matricxon/app/schemas/chat.py's ChatOptions field for field.
    """
    options = {
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "top_k": params["top_k"],
        "repeat_penalty": params["repeat_penalty"],
        "num_ctx": params["num_ctx"],
        "num_predict": params["num_predict"],
    }
    if params.get("seed", -1) != -1:
        options["seed"] = params["seed"]

    payload = {"model": model, "messages": messages, "options": options, "stream": True}
    add_tools_to_payload(payload, params)
    sink = sink_from(params)
    # The reply's id (from reply_generation_service): lets stop_model cancel just this reply without unloading the model
    # for others.
    if params.get("request_id"):
        payload["request_id"] = params["request_id"]
    if params.get("conversation_id"):  # tags the cached prompt, so a deleted chat's cache can be dropped
        payload["cache_tag"] = params["conversation_id"]

    # Filled by _raw_content_chunks_from (overwritten each attempt, as stream_with_failover may call it more than once)
    # and
    # read when the stream ends, to attach telemetry to the span. See ollama_client.chat_stream's call_state: the span
    # wraps the whole failover loop, so plain locals won't do.
    call_state: dict = {"attempts": 0}
    # Cluster: a chat's processed state lives in RAM on one node. Send it there, or, when it must go elsewhere
    # (that node busy or down), have that node write it to the shared disk first.
    chat_tag = params.get("conversation_id")
    holder, route = await matricxon_pool.route_hints(chat_tag, model)  # + what each node reports
    moved: dict = {}

    async def _raw_content_chunks_from(host: str) -> AsyncGenerator[str, None]:
        call_state["host"] = host
        call_state["attempts"] += 1
        matricxon_pool.note_model_used(host, model)
        if holder and holder != host and not moved:
            moved["spilled"] = await matricxon_pool.spill_chat(holder, chat_tag)
            logger.info(
                "chat %s moves from %s to %s (%s cached conversation(s) spilled)",
                chat_tag,
                holder,
                host,
                moved["spilled"],
            )
        async with matricxon_http.client(timeout=_CHAT_TIMEOUT) as client:
            async with client.stream("POST", f"{host}/api/chat", json=payload) as resp:
                if resp.is_error:
                    # Read the body (Matricxon's `{"error": "..."}`, e.g. InsufficientMemoryError's "not enough memory
                    # to load
                    # ...: need ~12.9GB, only 10.9GB available") before the `async with` closes the stream:
                    # httpx.HTTPStatusError.__str__
                    # omits the body and would drop the one part an admin can act on.
                    body = await resp.aread()
                    raise MatricxonError(_error_detail_from_body(body, resp.status_code))
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if sink:
                        sink.add_from_chunk(data)
                    if data.get("done"):
                        # Matricxon's server-side timing/token counts, present only on this final line
                        # (telemetry_exporter turns them into a TelemetryEvent row).
                        call_state["done_payload"] = data
                        break
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        yield chunk

    # record_exception/set_status_on_exception off (as ollama_client.chat_stream): their defaults mark the span
    # ERROR on any exception, including the normal asyncio.CancelledError from "stop". Status is set explicitly below.
    with matricxon_telemetry.get_tracer().start_as_current_span(
        "matricxon.chat", kind=SpanKind.CLIENT, record_exception=False, set_status_on_exception=False
    ) as span:
        span.set_attribute("gen_ai.request.model", model)
        try:
            async for chunk in matricxon_pool.stream_with_failover(_raw_content_chunks_from, prefer=holder, info=route):
                yield chunk
        except (httpx.HTTPError, MatricxonError) as exc:
            matricxon_telemetry.record_chat_error(span, call_state.get("host", ""), call_state["attempts"], exc)
            # A MatricxonError here already has its real message (e.g. the memory-refusal detail) from
            # _raw_content_chunks_from; re-wrapping would bury it under "Chat request failed:...".
            if isinstance(exc, MatricxonError):
                raise
            raise MatricxonError(f"Chat request failed: {exc}") from exc
        else:
            matricxon_telemetry.record_chat_success(
                span, call_state.get("host", ""), call_state["attempts"], call_state.get("done_payload", {})
            )


async def chat_once(model: str, messages: list[dict], params: dict | None = None) -> str:
    """Non-streaming helper for one-off internal calls (e.g. title generation); see ollama_client.chat_once."""
    params = params or {
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "num_ctx": 2048,
        "num_predict": 32,
        "seed": -1,
    }
    text = ""
    async for chunk in chat_stream(model, messages, params):
        text += chunk
    return text


async def stop_model(model: str, request_id: str | None = None) -> None:
    """Stops a reply: an empty `keep_alive: 0` call (../matricxon/app/schemas/chat.py ChatRequest.is_unload_call).
    With `request_id` (as sent by chat_stream) only that reply is cancelled and the model stays loaded; without
    it, the old "unload now" (the wire shape ollama_client.stop_model sends). Sent to every configured host; see that
    function.
    """
    body: dict = {"model": model, "messages": [], "keep_alive": 0}
    if request_id is not None:
        body["request_id"] = request_id
    for host in matricxon_pool.get_effective_hosts():
        try:
            async with matricxon_http.client(timeout=httpx.Timeout(5.0, connect=2.0)) as client:
                await client.post(f"{host}/api/chat", json=body)
        except httpx.HTTPError:
            pass
