"""
Thin async wrapper around Matricxon's local HTTP API — the Matricxon-flavored analogue of
app/services/ollama_client.py, deliberately a separate module rather than reusing that one: the two backends are
unrelated processes an admin configures independently (see app.services.engine_service), and keeping their HTTP
plumbing apart means neither can accidentally leak the other's failure/cooldown state or error type. Matricxon's
own HTTP API was built deliberately Ollama-shaped (../matricxon/app/main.py's own docstring calls it "an
Ollama-API-compatible inference server") — matched here endpoint-by-endpoint against its real router/schema code
(tags_router.py, chat_router.py, embeddings_router.py), not assumed from Ollama's docs.

Known gaps versus ollama_client.py, deliberate simplifications rather than oversights:
  - No "thinking"/reasoning-model stripping (app.services.ollama_thinking) — Matricxon only implements the
    mistral3/bert/nomic-bert architectures today (see ../matricxon/app/architectures/registry.py), none of which
    are thinking-capable, so there is nothing to strip yet. This is also why chat_stream below doesn't need
    ollama_client.chat_stream's extra nested _content_stream() generator — that exists purely to let that
    thinking-check pick which generator to iterate, which has nothing to mirror here.
  - Embeddings use Matricxon's own singular `/api/embeddings` (not Ollama's `/api/embed`) — same request/response
    shape otherwise (confirmed against ../matricxon/app/schemas/embeddings.py).

OpenTelemetry span instrumentation (see app.services.matricxon_telemetry for the attribute-shaping logic, kept
out of this file the same way ollama_client.py keeps it out in app.services.ollama_telemetry) mirrors
ollama_client.py's own chat_stream/embed spans field-for-field — confirmed against ../matricxon/app/routers/
chat_router.py directly that its done-line payload uses the identical prompt_eval_count/eval_count/
load_duration/eval_duration/total_duration field names Ollama's own does, so no separate attribute mapping was
needed."""

import json
import logging
from collections.abc import AsyncGenerator

import httpx
from opentelemetry.trace import SpanKind, Status, StatusCode

from app.config import EMBEDDING_NUM_CTX
from app.services import matricxon_pool, matricxon_telemetry

logger = logging.getLogger("llama_chat")

_REQUEST_TIMEOUT = httpx.Timeout(120.0, connect=5.0)
_CHAT_TIMEOUT = httpx.Timeout(None, connect=5.0)


class MatricxonError(Exception):
    """Raised when Matricxon is unreachable or returns an error — mirrors app.services.ollama_client.OllamaError's
    role, kept as its own type (not a shared base class) so a caller can never accidentally catch one engine's
    errors while believing it's handling the other's; app.services.inference_client normalizes both into one
    engine-agnostic InferenceError for callers that don't care which engine is active."""


async def list_models() -> list[dict]:
    """Returns the list of models currently pulled into Matricxon (what GET /api/tags shows) — same
    name/capabilities/details shape as Ollama's own /api/tags (see ../matricxon/app/routers/tags_router.py),
    which is what lets app.services.model_catalog_service build a catalog from either engine unchanged.

    Deliberately never cached, unlike get_capabilities below: callers need installed state genuinely fresh, or
    they'd reintroduce a "just pulled/uninstalled it, but the UI hasn't caught up" bug."""
    host = matricxon_pool.pick_host()
    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
        try:
            resp = await client.get(f"{host}/api/tags")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise MatricxonError(f"Could not reach Matricxon at {host}: {exc}") from exc
    return resp.json().get("models", [])


async def get_capabilities() -> dict:
    """Matricxon's own real, current support surface — GET /api/health (see
    ../matricxon/app/routers/health_router.py), a genuine Matricxon-only extension with no Ollama equivalent,
    added specifically so a caller can ask "can you run this?" directly instead of hand-maintaining its own
    separate, driftable copy of Matricxon's architecture/quantization registries (see
    ../matricxon/ROADMAP.md's "Expose supported architectures/quantizations" entry, and
    app.services.matricxon_support_checker.MatricxonSupportChecker, the only caller). Returns
    `{"supported_architectures": [...], "supported_quantizations": [...]}` — every `general.architecture` string
    and every GGML quantization type name Matricxon can currently load, regardless of which engine is actually
    active right now (see engine_service.current_engine) — this is asked about Matricxon specifically, not
    "whichever engine is active", so unlike every other function in this file it's never routed through
    app.services.inference_client's active-engine dispatch.

    Deliberately uncached — never fetch this once and reuse it across requests. This used to be cached for
    30s on the (reasonable-sounding) theory that "what Matricxon can run at all only changes on a Matricxon
    upgrade+restart", but that's exactly the case that broke live, 2026-09-22: Matricxon's own architecture
    registry can gain a new entry (granite/granitemoe/nemotron_h all landed in a single session) without
    pAIring's own process ever restarting, and a stale cached answer kept showing "not supported" for
    something that had genuinely just become supported, with no way to force a refresh short of restarting
    pAIring itself. /api/health is cheap on Matricxon's own side (HealthRequestHandler.handle just returns two
    in-memory Python lists — no file I/O, no model access, see ../matricxon/app/routers/health_router.py), so
    the redundancy this cache used to avoid (three separate catalog builders each asking within the same page
    load) isn't worth trading away real-time accuracy for."""
    host = matricxon_pool.pick_host()
    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
        try:
            resp = await client.get(f"{host}/api/health")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise MatricxonError(f"Could not reach Matricxon at {host}: {exc}") from exc
    return resp.json()


async def embed(text: str, model: str) -> list[float]:
    """Turns `text` into an embedding vector via Matricxon's own POST /api/embeddings (singular — see this
    module's own docstring), using `model` (the admin's configured default — see
    app.services.model_catalog_service.get_default_embedding_model, resolved by every caller before this is
    reached). No retry-on-cold-start (see app.services.ollama_client.embed's own docstring for why Ollama needs
    one): Matricxon's ModelManager loads a model synchronously inside the request itself (see
    ../matricxon/app/routers/embeddings_router.py), so there is no analogous "dispatched before the backend was
    ready" race to absorb — hence also just one span attempt_count always, unlike ollama_client.embed's retry loop."""
    host = matricxon_pool.pick_host()
    with matricxon_telemetry.get_tracer().start_as_current_span(
        "matricxon.embed", kind=SpanKind.CLIENT, record_exception=False, set_status_on_exception=False
    ) as span:
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("server.address", host)
        span.set_attribute("ollama.attempt_count", 1)
        async with matricxon_pool.track_request(host):
            async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
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
    """Pulls Matricxon's own `{"error": "..."}` detail out of an error response body — falls back to a generic
    message only when the body isn't the shape Matricxon's own routers always send (../matricxon/app/routers/
    chat_router.py's error responses, and every other router's, are this same shape)."""
    try:
        detail = json.loads(body).get("error")
    except (json.JSONDecodeError, AttributeError):
        detail = None
    return detail or f"Matricxon returned HTTP {status_code} with no error detail."


async def chat_stream(model: str, messages: list[dict], params: dict) -> AsyncGenerator[str, None]:
    """Streams a chat completion from Matricxon, yielding text chunks as they're generated — see
    app.services.ollama_client.chat_stream's own docstring for the shared request/response contract (`messages`
    role/content dicts in, sub-word text chunks out); matched field-for-field against
    ../matricxon/app/schemas/chat.py's ChatOptions, which mirrors this same options dict exactly."""
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

    # Populated by _raw_content_chunks_from below (overwritten on every attempt, since stream_with_failover can
    # call it more than once) and read back once the stream finishes, to attach telemetry to the span below —
    # see ollama_client.chat_stream's identical call_state for why this can't just be local variables closed
    # over normally (the span has to wrap the whole failover loop, not just one attempt).
    call_state: dict = {"attempts": 0}

    async def _raw_content_chunks_from(host: str) -> AsyncGenerator[str, None]:
        call_state["host"] = host
        call_state["attempts"] += 1
        async with httpx.AsyncClient(timeout=_CHAT_TIMEOUT) as client:
            async with client.stream("POST", f"{host}/api/chat", json=payload) as resp:
                if resp.is_error:
                    # Read the body (Matricxon's own {"error": "..."} detail, e.g. InsufficientMemoryError's real
                    # "not enough memory to load ...: need ~12.9GB, only 10.9GB available" reason) before the
                    # `async with` above closes the stream on the way out — httpx.HTTPStatusError.__str__ never
                    # carries the response body, only status/url, so raising that here would silently drop the one
                    # part of this error an admin can actually act on.
                    body = await resp.aread()
                    raise MatricxonError(_error_detail_from_body(body, resp.status_code))
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("done"):
                        # Matricxon's own server-side timing/token counts, only ever present on this final line —
                        # see app.services.telemetry_exporter for how these become a TelemetryEvent row.
                        call_state["done_payload"] = data
                        break
                    chunk = data.get("message", {}).get("content", "")
                    if chunk:
                        yield chunk

    # record_exception/set_status_on_exception off — see ollama_client.chat_stream's identical comment: their
    # defaults would auto-mark this span ERROR on *any* propagating exception, including asyncio.CancelledError
    # from a user clicking "stop", a normal path, not an error. Status is only ever set explicitly, below.
    with matricxon_telemetry.get_tracer().start_as_current_span(
        "matricxon.chat", kind=SpanKind.CLIENT, record_exception=False, set_status_on_exception=False
    ) as span:
        span.set_attribute("gen_ai.request.model", model)
        try:
            async for chunk in matricxon_pool.stream_with_failover(_raw_content_chunks_from):
                yield chunk
        except (httpx.HTTPError, MatricxonError) as exc:
            matricxon_telemetry.record_chat_error(span, call_state.get("host", ""), call_state["attempts"], exc)
            # A MatricxonError here already came from _raw_content_chunks_from above with its own real message
            # (e.g. the memory-refusal detail) — re-wrapping it would only bury that behind a second, less
            # useful "Chat request failed: ..." layer, the exact loss of detail this error type was added to fix.
            if isinstance(exc, MatricxonError):
                raise
            raise MatricxonError(f"Chat request failed: {exc}") from exc
        else:
            matricxon_telemetry.record_chat_success(
                span, call_state.get("host", ""), call_state["attempts"], call_state.get("done_payload", {})
            )


async def chat_once(model: str, messages: list[dict], params: dict | None = None) -> str:
    """Non-streaming helper for one-off internal calls (e.g. title generation) — see
    app.services.ollama_client.chat_once's identical purpose."""
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
    """Forces Matricxon to unload `model` right now — `keep_alive: 0` with no messages, which
    ../matricxon/app/schemas/chat.py's ChatRequest.is_unload_call() recognizes as an explicit unload, the exact
    same wire shape app.services.ollama_client.stop_model already sends to Ollama. Sent to every configured host;
    see that function's own docstring for the full reasoning, identical here."""
    for host in matricxon_pool.get_effective_hosts():
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=2.0)) as client:
                await client.post(f"{host}/api/chat", json={"model": model, "messages": [], "keep_alive": 0})
        except httpx.HTTPError:
            pass
