"""Thin async wrapper around Ollama's local HTTP API.

Knows nothing of FastAPI, SQLite or the web UI, so Ollama could be swapped for another local server by
rewriting this file. Every real call is an OpenTelemetry span for the Telemetry page; attribute
shaping lives in ollama_telemetry.

Ollama's docs: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

import json
import logging
from collections.abc import AsyncGenerator

import httpx
from opentelemetry.trace import SpanKind, Status, StatusCode

from app.config import EMBEDDING_NUM_CTX
from app.services import ollama_pool, ollama_telemetry
from app.services.ollama_thinking import model_supports_thinking, strip_inline_thinking
from app.services.tool_call_sink import add_tools_to_payload, sink_from

logger = logging.getLogger("llama_chat")

# For a quick, non-generation call (list/embed).
_REQUEST_TIMEOUT = httpx.Timeout(120.0, connect=5.0)
# chat_stream's own, unbounded; sharing _REQUEST_TIMEOUT silently undercut reply_generation_service's timeout.
_CHAT_TIMEOUT = httpx.Timeout(None, connect=5.0)


class OllamaError(Exception):
    """Raised when Ollama is unreachable or errors, for clean HTTP errors instead of stack traces."""


async def list_models() -> list[dict]:
    """Models pulled into Ollama (`ollama list`), for the Settings picker. Picks a host like chat/embed
    (ollama_pool); in remote mode with several hosts the catalogs may differ (as in ollama_admin's pull/delete).
    """
    host = ollama_pool.pick_host()
    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
        try:
            resp = await client.get(f"{host}/api/tags")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(f"Could not reach Ollama at {host}: {exc}") from exc
    return resp.json().get("models", [])


# One retry for a failed embedding. The *first* request after the model idled out (OLLAMA_KEEP_ALIVE, default
# 5 minutes) reliably fails with a 500 (Ollama dispatches before llama-server is ready); later ones succeed.
# One retry absorbs that cold start.
_EMBED_MAX_ATTEMPTS = 2


async def embed(text: str, model: str) -> list[float]:
    """Embeds `text` with `model` (the admin default, resolved by callers via
    model_catalog_service.get_default_embedding_model).
    Used for document ingest (document_ingest.py) and the live question (document_retrieval.py).

    Picks a host once (ollama_pool), then retries the *same* host up to _EMBED_MAX_ATTEMPTS times for the cold
    start (a warm-up cost, not "host down"), so it doesn't fail over mid-call like chat_stream. A host is marked
    failed (cooldown) only after every attempt fails.
    """
    host = ollama_pool.pick_host()
    last_error: Exception | None = None
    tracer = ollama_telemetry.get_tracer()
    # record_exception/set_status_on_exception off; see chat_stream.
    with tracer.start_as_current_span(
        "ollama.embed", kind=SpanKind.CLIENT, record_exception=False, set_status_on_exception=False
    ) as span:
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("server.address", host)
        async with ollama_pool.track_request(host):
            for attempt in range(1, _EMBED_MAX_ATTEMPTS + 1):
                async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
                    try:
                        resp = await client.post(
                            f"{host}/api/embeddings",
                            json={
                                "model": model,
                                "prompt": text,
                                "options": {"num_ctx": EMBEDDING_NUM_CTX},
                            },
                        )
                        resp.raise_for_status()
                        ollama_pool.mark_recovered(host)
                        span.set_attribute("ollama.attempt_count", attempt)
                        return resp.json()["embedding"]
                    except httpx.HTTPError as exc:
                        last_error = exc
                        logger.warning(
                            "Embedding request failed (attempt %d/%d): %s",
                            attempt,
                            _EMBED_MAX_ATTEMPTS,
                            exc,
                        )
            ollama_pool.mark_failure(host)
            span.set_attribute("ollama.attempt_count", _EMBED_MAX_ATTEMPTS)
            span.set_status(Status(StatusCode.ERROR, str(last_error)))
    raise OllamaError(f"Embedding request failed: {last_error}") from last_error


async def chat_stream(
    model: str,
    messages: list[dict],
    params: dict,
) -> AsyncGenerator[str, None]:
    """Streams a chat completion as text chunks (sub-word tokens, joined back-to-back on the frontend).

    `messages` is oldest-first {"role", "content"} dicts (the Message row shape, so history forwards as is).

    `params` is a GenerationParams-shaped dict (app/schemas.py) passed to Ollama's `options`
    (temperature/top_p/top_k/...).

    A "thinking" model's reasoning (ollama_thinking model_supports_thinking/strip_inline_thinking) is always
    filtered out: no caller wants the inner monologue in a reply.
    """
    options = {
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "top_k": params["top_k"],
        "repeat_penalty": params["repeat_penalty"],
        "num_ctx": params["num_ctx"],
        "num_predict": params["num_predict"],
    }
    # A seed of -1 means "random" in our schema; Ollama wants the key omitted.
    if params.get("seed", -1) != -1:
        options["seed"] = params["seed"]

    payload = {
        "model": model,
        "messages": messages,
        "options": options,
        "stream": True,
    }
    add_tools_to_payload(payload, params)
    sink = sink_from(params)

    # Filled by _raw_content_chunks_from (overwritten each attempt; stream_with_failover may call it more than once)
    # and read when the stream ends to attach telemetry to the span _content_stream opens. That span starts lazily
    # on first pull, not at chat_stream's call, so it can't wrap this outer call.
    call_state: dict = {"attempts": 0}

    async def _raw_content_chunks_from(host: str) -> AsyncGenerator[str, None]:
        # No try/except on purpose: an httpx.HTTPError propagates to ollama_pool.stream_with_failover, which decides
        # whether a retry on another host is safe (only before any chunk is yielded).
        call_state["host"] = host
        call_state["attempts"] += 1
        async with httpx.AsyncClient(timeout=_CHAT_TIMEOUT) as client:
            async with client.stream("POST", f"{host}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                # Ollama streams NDJSON: one JSON object per line with the next chunk of the reply.
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if sink:
                        sink.add_from_chunk(data)
                    if data.get("done"):
                        # Ollama's server-side timing/token counts, only on this final line (telemetry_exporter ->
                        # TelemetryEvent row).
                        call_state["done_payload"] = data
                        break
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        yield chunk

    async def _content_stream() -> AsyncGenerator[str, None]:
        # record_exception/set_status_on_exception off: defaults would mark the span ERROR on any exception, including
        # asyncio.CancelledError from "stop" (normal). Status is set explicitly in the except block below.
        with ollama_telemetry.get_tracer().start_as_current_span(
            "ollama.chat", kind=SpanKind.CLIENT, record_exception=False, set_status_on_exception=False
        ) as span:
            span.set_attribute("gen_ai.request.model", model)
            try:
                async for chunk in ollama_pool.stream_with_failover(_raw_content_chunks_from):
                    yield chunk
            except httpx.HTTPError as exc:
                ollama_telemetry.record_chat_error(span, call_state.get("host", ""), call_state["attempts"], exc)
                raise OllamaError(f"Chat request failed: {exc}") from exc
            else:
                ollama_telemetry.record_chat_success(
                    span, call_state.get("host", ""), call_state["attempts"], call_state.get("done_payload", {})
                )

    if await model_supports_thinking(model):
        async for chunk in strip_inline_thinking(_content_stream()):
            yield chunk
    else:
        async for chunk in _content_stream():
            yield chunk


async def chat_once(model: str, messages: list[dict], params: dict | None = None) -> str:
    """Non-streaming helper for one-off internal calls (e.g. "smart" title generation,
    chat_service._maybe_generate_title_with_model) wanting final text only. Built on chat_stream, so "thinking" is
    filtered too.
    """
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


async def stop_model(model: str) -> None:
    """Forces Ollama to unload `model` now (`keep_alive: 0`, no messages; like `ollama stop <model>`): cleanup after
    this app gave up on a generation (reply timeout or cancel; reply_generation_service/reply_termination_service),
    since closing our connection doesn't reliably interrupt llama.cpp compute (prompt/image processing seemingly
    ignores disconnects).

    Sent to every configured host, not just the one that served the request (stream_with_failover doesn't expose
    it): a no-op where the model isn't running, and an accepted best-effort limit for multi-host setups. Failures
    are swallowed: cleanup must not mask the real termination reason. No telemetry span: a cleanup fan-out, not
    a request for the dashboard.
    """
    for host in ollama_pool.get_effective_hosts():
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=2.0)) as client:
                await client.post(f"{host}/api/chat", json={"model": model, "messages": [], "keep_alive": 0})
        except httpx.HTTPError:
            pass
