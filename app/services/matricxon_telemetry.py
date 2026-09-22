"""OpenTelemetry SDK bootstrap for the Telemetry page's Matricxon
instrumentation — the Matricxon-flavored twin of app.services.ollama_telemetry (see app.services.matricxon_client
for the actual spans, app.services.telemetry_exporter for where both engines' spans end up, shared unmodified).
Kept as its own module rather than parameterizing ollama_telemetry.py, matching how this codebase already keeps
ollama_client.py/matricxon_client.py, ollama_pool.py/matricxon_pool.py, etc. as separate same-shaped files rather
than one shared abstraction — see this module's own functions for the (deliberately identical) reasoning behind
each one, copied from ollama_telemetry.py's own docstrings.

A separate TracerProvider (not a shared one, and not just a second BatchSpanProcessor on Ollama's) so an admin
disabling/removing one engine's instrumentation later can't accidentally affect the other's — and so each
engine's own get_tracer() stays a simple, independent call with no engine-name branching anywhere in either
engine's own client module."""

import asyncio
import logging

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Status, StatusCode

from app.config import INSTANCE_INDEX
from app.services.telemetry_exporter import DBSpanExporter, capture_main_loop

# Same field names Matricxon's own /api/chat done-line already uses (confirmed against
# ../matricxon/app/routers/chat_router.py directly, not assumed from Ollama's shape) — matricxon_client.py's own
# docstring already establishes it was "matched field-for-field against Ollama", so this mapping needs no
# separate Matricxon-specific version. Matricxon's done line never carries prompt_eval_duration (no separate
# prompt-processing timer exists there yet) — harmless, same as Ollama's own embed() calls never populating any
# of these: record_chat_success below only ever sets what's actually present.
_CHAT_DURATION_FIELDS = (
    ("total_duration", "ollama.total_duration_ns"),
    ("load_duration", "ollama.load_duration_ns"),
    ("prompt_eval_duration", "ollama.prompt_eval_duration_ns"),
    ("eval_duration", "ollama.eval_duration_ns"),
)

logger = logging.getLogger("llama_chat")

_SCHEDULE_DELAY_MILLIS = 2000

_tracer_provider: TracerProvider | None = None


def init_matricxon_telemetry() -> None:
    """Called once per process from startup_service.run_startup_tasks()
    — on every instance (not primary-gated), since each instance routes
    its own Matricxon calls independently and each needs its own tracer."""
    global _tracer_provider
    capture_main_loop()
    resource = Resource.create({"service.name": "pairing", "service.instance.id": str(INSTANCE_INDEX)})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(DBSpanExporter(), schedule_delay_millis=_SCHEDULE_DELAY_MILLIS))
    _tracer_provider = provider
    logger.info("Matricxon telemetry initialized (instance %d).", INSTANCE_INDEX)


def get_tracer() -> trace.Tracer:
    """Safe to call before init_matricxon_telemetry() has run (e.g. in any test that never calls it) — falls
    back to the OTel API's own no-op tracer rather than touching the global registry (see this module's own
    docstring for why the real provider is never installed there)."""
    if _tracer_provider is not None:
        return _tracer_provider.get_tracer("pairing.matricxon")
    return trace.get_tracer("pairing.matricxon")


def record_chat_success(span: Span, host: str, attempts: int, done_payload: dict) -> None:
    """Fills in a "matricxon.chat" span's outcome once chat_stream's NDJSON loop finishes normally —
    done_payload is whatever Matricxon's final `done:true` line carried (empty if a caller closed the generator
    before that line ever arrived, e.g. a cancelled reply)."""
    span.set_attribute("server.address", host)
    span.set_attribute("ollama.attempt_count", attempts)
    if "prompt_eval_count" in done_payload:
        span.set_attribute("gen_ai.usage.input_tokens", done_payload["prompt_eval_count"])
    if "eval_count" in done_payload:
        span.set_attribute("gen_ai.usage.output_tokens", done_payload["eval_count"])
    for matricxon_key, attr_name in _CHAT_DURATION_FIELDS:
        if matricxon_key in done_payload:
            span.set_attribute(attr_name, done_payload[matricxon_key])


def record_chat_error(span: Span, host: str, attempts: int, exc: Exception) -> None:
    span.set_attribute("server.address", host)
    span.set_attribute("ollama.attempt_count", attempts)
    span.set_status(Status(StatusCode.ERROR, str(exc)))


async def shutdown_matricxon_telemetry() -> None:
    """Called from startup_service.run_shutdown_tasks(). See
    ollama_telemetry.shutdown_ollama_telemetry's own docstring for why this hops onto a worker thread rather than
    calling TracerProvider.shutdown() directly on the main event loop — identical deadlock risk, identical fix."""
    if _tracer_provider is not None:
        await asyncio.to_thread(_tracer_provider.shutdown)
