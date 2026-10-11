"""OpenTelemetry SDK bootstrap for the Telemetry page's Ollama instrumentation (spans: ollama_client;
destination: telemetry_exporter).

Never calls opentelemetry.trace.set_tracer_provider: that writes *process-global* state, and the whole
test suite runs in one pytest process, so a test configuring the real SDK that way would leak into later
tests (the global provider is effectively set-once), making results depend on collection order. This module
keeps its own TracerProvider and hands it out through get_tracer; callers (ollama_client.py) use this
wrapper, never the global accessor.
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

# Ollama's field names on the final NDJSON `done:true` line of /api/chat mapped to span attributes; kept here
# so ollama_client.py stays a thin HTTP wrapper.
_CHAT_DURATION_FIELDS = (
    ("total_duration", "ollama.total_duration_ns"),
    ("load_duration", "ollama.load_duration_ns"),
    ("prompt_eval_duration", "ollama.prompt_eval_duration_ns"),
    ("eval_duration", "ollama.eval_duration_ns"),
)

logger = logging.getLogger("llama_chat")

# Well under BatchSpanProcessor's 5s default: a single self-hosted deployment has no batching reason to hold a
# finished span that long, and a multi-second lag before a chat shows on the Telemetry page looks "broken".
_SCHEDULE_DELAY_MILLIS = 2000

_tracer_provider: TracerProvider | None = None


def init_ollama_telemetry() -> None:
    """Called once per process from startup_service.run_startup_tasks, on every instance (not primary-gated): each
    routes its own Ollama calls and needs its own tracer.
    """
    global _tracer_provider
    capture_main_loop()
    resource = Resource.create({"service.name": "pairing", "service.instance.id": str(INSTANCE_INDEX)})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(DBSpanExporter(), schedule_delay_millis=_SCHEDULE_DELAY_MILLIS))
    _tracer_provider = provider
    logger.info("Ollama telemetry initialized (instance %d).", INSTANCE_INDEX)


def get_tracer() -> trace.Tracer:
    """Safe before init_ollama_telemetry (e.g. in tests): falls back to the OTel API's no-op tracer without touching
    the global registry."""
    if _tracer_provider is not None:
        return _tracer_provider.get_tracer("pairing.ollama")
    return trace.get_tracer("pairing.ollama")


def record_chat_success(span: Span, host: str, attempts: int, done_payload: dict) -> None:
    """Fills an "ollama.chat" span's outcome once chat_stream's NDJSON loop ends normally. done_payload is Ollama's
    final `done:true` line (empty if the generator closed first, e.g. a cancelled reply).
    """
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
    """Called from startup_service.run_shutdown_tasks. Must not call TracerProvider.shutdown on the main loop's
    thread: it blocks joining BatchSpanProcessor's worker, which is blocked handing its final export back to
    this loop via run_coroutine_threadsafe, a deadlock bounded only by telemetry_exporter's export timeout.
    A worker thread keeps the loop free to run that export's DB write.
    """
    if _tracer_provider is not None:
        await asyncio.to_thread(_tracer_provider.shutdown)
