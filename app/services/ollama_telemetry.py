"""OpenTelemetry SDK bootstrap for the Telemetry page's Ollama
instrumentation (see app.services.ollama_client for the actual spans,
app.services.telemetry_exporter for where they end up).

Deliberately never calls opentelemetry.trace.set_tracer_provider() —
that writes to *process-global* state, and this app's whole test suite
runs in one pytest process. A test that configured the real SDK that
way would leak into every other test after it (the OTel API documents
the global provider as effectively set-once), making pass/fail depend
on test collection order. Instead this module keeps its own
module-level TracerProvider and hands it out through get_tracer() —
callers (ollama_client.py) always go through this wrapper, never the
OTel API's global accessor, so the global registry is never touched and
nothing here can bleed across tests.
"""

import asyncio
import logging

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Status, StatusCode

from app.config import INSTANCE_INDEX
from app.services.telemetry_exporter import DBSpanExporter, capture_main_loop

# Maps Ollama's own field names, on the final NDJSON `done:true` line of
# a /api/chat response, to the span attribute each becomes — kept here
# (not in ollama_client.py) so that file stays a thin HTTP wrapper, not
# also the place span/attribute shaping logic lives.
_CHAT_DURATION_FIELDS = (
    ("total_duration", "ollama.total_duration_ns"),
    ("load_duration", "ollama.load_duration_ns"),
    ("prompt_eval_duration", "ollama.prompt_eval_duration_ns"),
    ("eval_duration", "ollama.eval_duration_ns"),
)

logger = logging.getLogger("llama_chat")

# Well under BatchSpanProcessor's 5s default — for a single self-hosted
# deployment there's no batching-volume reason to sit on a finished span
# that long, and a multi-second lag before a chat shows up on the
# Telemetry page reads as "broken" during live verification.
_SCHEDULE_DELAY_MILLIS = 2000

_tracer_provider: TracerProvider | None = None


def init_ollama_telemetry() -> None:
    """Called once per process from startup_service.run_startup_tasks()
    — on every instance (not primary-gated), since each instance routes
    its own Ollama calls independently and each needs its own tracer."""
    global _tracer_provider
    capture_main_loop()
    resource = Resource.create({"service.name": "pairing", "service.instance.id": str(INSTANCE_INDEX)})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(DBSpanExporter(), schedule_delay_millis=_SCHEDULE_DELAY_MILLIS))
    _tracer_provider = provider
    logger.info("Ollama telemetry initialized (instance %d).", INSTANCE_INDEX)


def get_tracer() -> trace.Tracer:
    """Safe to call before init_ollama_telemetry() has run (e.g. in any
    test that never calls it) — falls back to the OTel API's own no-op
    tracer rather than touching the global registry."""
    if _tracer_provider is not None:
        return _tracer_provider.get_tracer("pairing.ollama")
    return trace.get_tracer("pairing.ollama")


def record_chat_success(span: Span, host: str, attempts: int, done_payload: dict) -> None:
    """Fills in an "ollama.chat" span's outcome once chat_stream's NDJSON
    loop finishes normally — done_payload is whatever Ollama's final
    `done:true` line carried (empty if a caller closed the generator
    before that line ever arrived, e.g. a cancelled reply)."""
    span.set_attribute("server.address", host)
    span.set_attribute("ollama.attempt_count", attempts)
    if "prompt_eval_count" in done_payload:
        span.set_attribute("gen_ai.usage.input_tokens", done_payload["prompt_eval_count"])
    if "eval_count" in done_payload:
        span.set_attribute("gen_ai.usage.output_tokens", done_payload["eval_count"])
    for ollama_key, attr_name in _CHAT_DURATION_FIELDS:
        if ollama_key in done_payload:
            span.set_attribute(attr_name, done_payload[ollama_key])


def record_chat_error(span: Span, host: str, attempts: int, exc: Exception) -> None:
    span.set_attribute("server.address", host)
    span.set_attribute("ollama.attempt_count", attempts)
    span.set_status(Status(StatusCode.ERROR, str(exc)))


async def shutdown_ollama_telemetry() -> None:
    """Called from startup_service.run_shutdown_tasks(). Must not call
    TracerProvider.shutdown() directly on this (the main event loop's)
    thread — it blocks joining BatchSpanProcessor's worker thread, which
    is itself blocked handing its final export() back to this same loop
    via run_coroutine_threadsafe — a deadlock, bounded only by
    telemetry_exporter's own export timeout. Hopping the blocking call
    onto a worker thread keeps the loop free to actually run that final
    export's DB write."""
    if _tracer_provider is not None:
        await asyncio.to_thread(_tracer_provider.shutdown)
