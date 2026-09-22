"""
Thin async wrapper around Ollama's local HTTP API.

Nothing in this file knows about FastAPI, SQLite, or the web UI — it only knows how to talk to Ollama. Keeping that
boundary clean means Ollama could be swapped for another local-inference server later by rewriting just this one
file (see ROADMAP.md). Every real call is wrapped in an OpenTelemetry span for the Telemetry page — attribute-shaping
logic lives in app.services.ollama_telemetry, not here, to keep this file focused on Ollama's own HTTP shape.

Ollama's own docs: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

import json
import logging
from collections.abc import AsyncGenerator

import httpx
from opentelemetry.trace import SpanKind, Status, StatusCode

from app.config import EMBEDDING_NUM_CTX
from app.services import ollama_pool, ollama_telemetry
from app.services.ollama_thinking import model_supports_thinking, strip_inline_thinking

logger = logging.getLogger("llama_chat")

# For a quick, non-generation call (list/embed).
_REQUEST_TIMEOUT = httpx.Timeout(120.0, connect=5.0)
# chat_stream's own — unbounded; sharing _REQUEST_TIMEOUT used to silently undercut reply_generation_service's timeout.
_CHAT_TIMEOUT = httpx.Timeout(None, connect=5.0)


class OllamaError(Exception):
    """Raised when Ollama is unreachable or returns an error, so routers can turn it into a clean HTTP error
    instead of a raw stack trace."""


async def list_models() -> list[dict]:
    """Returns the list of models currently pulled into Ollama (what `ollama list` shows), used to populate the
    model picker in Settings. Picks a host from the pool the same way chat/embed calls do (see
    app.services.ollama_pool) — in "remote" mode with more than one host configured, this doesn't guarantee every
    host has the same catalog; matches app.services.ollama_admin's own documented pull/delete limitation for the
    same reason."""
    host = ollama_pool.pick_host()
    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
        try:
            resp = await client.get(f"{host}/api/tags")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(f"Could not reach Ollama at {host}: {exc}") from exc
    return resp.json().get("models", [])


# One retry for a failed embedding call. In practice, the *first* embedding request after the model has been idle
# long enough for Ollama to unload it (OLLAMA_KEEP_ALIVE, 5 minutes by default) reliably fails with a 500 — Ollama
# dispatches it before llama-server is actually ready — and every request after that succeeds normally once the
# model is warm. Retrying once transparently absorbs that cold-start hiccup instead of surfacing it as an error.
_EMBED_MAX_ATTEMPTS = 2


async def embed(text: str, model: str) -> list[float]:
    """Turns a piece of text into an embedding vector using `model` (the admin's configured default — see
    app.services.model_catalog_service.get_default_embedding_model, resolved by every caller before this is
    reached). Used both when ingesting documents (app/services/document_ingest.py) and when embedding the user's
    live question to find matching chunks (app/services/document_retrieval.py).

    Picks a host from the pool (see app/services/ollama_pool.py) once per call, then retries the *same* host up to
    _EMBED_MAX_ATTEMPTS times — that retry is for the cold-start hiccup documented below, a per-model/per-host
    warm-up cost, not a "this host is down" signal, so it doesn't fail over to a different host mid-call the way
    chat_stream does. A host is only marked failed (and put in cooldown for future calls) once every attempt
    against it has failed."""
    host = ollama_pool.pick_host()
    last_error: Exception | None = None
    tracer = ollama_telemetry.get_tracer()
    # record_exception/set_status_on_exception off — see chat_stream's own comment below on why.
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
    """Streams a chat completion from Ollama, yielding text chunks as they're generated (not whole words/sentences —
    Ollama streams sub-word tokens, so chunks are joined back-to-back on the frontend).

    `messages` is a list of {"role": ..., "content": ...} dicts, oldest first — the same shape Message rows are
    stored in, so callers can forward conversation history with no reshaping.

    `params` is a GenerationParams-shaped dict (see app/schemas.py); it's passed straight through to Ollama's
    `options` field, which is where Ollama expects temperature/top_p/top_k/etc to live.

    A "thinking"-capable model's raw reasoning (see app.services.ollama_thinking's model_supports_thinking/
    strip_inline_thinking) is always filtered out of what this yields — never surfaced as an option, since there's
    no situation where a caller actually wants a reply prefixed with the model's internal monologue.
    """
    options = {
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "top_k": params["top_k"],
        "repeat_penalty": params["repeat_penalty"],
        "num_ctx": params["num_ctx"],
        "num_predict": params["num_predict"],
    }
    # A seed of -1 means "random every time" in our own schema; Ollama
    # instead expects the key simply omitted for that behavior.
    if params.get("seed", -1) != -1:
        options["seed"] = params["seed"]

    payload = {
        "model": model,
        "messages": messages,
        "options": options,
        "stream": True,
    }

    # Populated by _raw_content_chunks_from below (overwritten on every attempt, since stream_with_failover can call
    # it more than once) and read back once the stream finishes, to attach telemetry to the span _content_stream
    # opens — that span has to start lazily, on first pull, not at chat_stream()'s own call time, which is why it
    # can't just wrap this whole outer call instead.
    call_state: dict = {"attempts": 0}

    async def _raw_content_chunks_from(host: str) -> AsyncGenerator[str, None]:
        # No try/except here on purpose: an httpx.HTTPError raised here propagates up through
        # ollama_pool.stream_with_failover, which is what actually decides whether it's safe to retry on a
        # different host (only before any chunk below has been yielded) — see that function's own docstring.
        call_state["host"] = host
        call_state["attempts"] += 1
        async with httpx.AsyncClient(timeout=_CHAT_TIMEOUT) as client:
            async with client.stream("POST", f"{host}/api/chat", json=payload) as resp:
                resp.raise_for_status()
                # Ollama streams one JSON object per line (NDJSON), each
                # containing the next chunk of the assistant's reply.
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("done"):
                        # Ollama's own server-side timing/token counts, only ever present on this final line — see
                        # app.services.telemetry_exporter for how these become a TelemetryEvent row.
                        call_state["done_payload"] = data
                        break
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        yield chunk

    async def _content_stream() -> AsyncGenerator[str, None]:
        # record_exception/set_status_on_exception off: their defaults would auto-mark this span ERROR on *any*
        # propagating exception, including asyncio.CancelledError from a user clicking "stop" — a normal, common
        # path, not an error. Status is only ever set explicitly, in the except block below.
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
    """Non-streaming helper for one-off internal calls (e.g. "smart" mode conversation-title generation — see
    app.services.chat_service._maybe_generate_title_with_model) where we just want the final text, not a
    live-typing effect. Built on chat_stream, so a "thinking"-capable model's raw reasoning is filtered out here
    too, the same as any other caller."""
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
    """Forces Ollama to unload `model` right now (`keep_alive: 0` with no messages — the same effect as running
    `ollama stop <model>` from the CLI) — belt-and-suspenders cleanup called after this app has already given up on
    a generation (a reply timeout, or a cancelled task — see reply_generation_service/reply_termination_service),
    since closing our own HTTP connection to Ollama doesn't reliably interrupt an in-progress llama.cpp compute
    phase (prompt/image processing especially doesn't appear to check for client disconnection at all).

    Sent to every configured host, not just whichever one actually served this request (stream_with_failover above
    never exposes that back to a caller) — a harmless no-op on any host not currently running this model, and an
    accepted best-effort limitation for a genuinely multi-host deployment, consistent with this codebase's other
    documented multi-instance limitations. Failures are swallowed on purpose — this is a cleanup nicety, never
    something that should mask the real termination reason a caller is already handling. Not instrumented with a
    telemetry span itself — it's cleanup fan-out, not a request whose latency/outcome belongs on the dashboard."""
    for host in ollama_pool.get_effective_hosts():
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=2.0)) as client:
                await client.post(f"{host}/api/chat", json={"model": model, "messages": [], "keep_alive": 0})
        except httpx.HTTPError:
            pass
