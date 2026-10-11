"""OpenTelemetry SDK bootstrap for the Telemetry page's Matricxon instrumentation: the twin of ollama_telemetry
(spans: matricxon_client; storage: telemetry_exporter, shared by both). A separate module rather than
parameterizing ollama_telemetry.py, as the codebase keeps ollama_*/matricxon_* files separate (reasoning
copied from ollama_telemetry's docstrings).

A separate TracerProvider (not shared, not a second BatchSpanProcessor on Ollama's) so disabling one
engine's instrumentation can't affect the other's, and each get_tracer stays independent with no
engine-name branching.
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

# Field names match Matricxon's /api/chat done-line (checked against../matricxon/app/routers/chat_router.py),
# which matricxon_client.py matches to Ollama's, so no Matricxon-specific mapping. Its done line has no
# prompt_eval_duration (no separate prompt timer yet); harmless, as record_chat_success sets only what's present.
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
    """Called once per process from startup_service.run_startup_tasks, on every instance (not primary-gated):
    each routes its own Matricxon calls and needs its own tracer.
    """
    global _tracer_provider
    capture_main_loop()
    resource = Resource.create({"service.name": "pairing", "service.instance.id": str(INSTANCE_INDEX)})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(DBSpanExporter(), schedule_delay_millis=_SCHEDULE_DELAY_MILLIS))
    _tracer_provider = provider
    logger.info("Matricxon telemetry initialized (instance %d).", INSTANCE_INDEX)


def get_tracer() -> trace.Tracer:
    """Safe before init_matricxon_telemetry (e.g. in tests): falls back to the OTel API's no-op tracer without
    touching the global registry (see module docstring).
    """
    if _tracer_provider is not None:
        return _tracer_provider.get_tracer("pairing.matricxon")
    return trace.get_tracer("pairing.matricxon")


def record_chat_success(span: Span, host: str, attempts: int, done_payload: dict) -> None:
    """Fills a "matricxon.chat" span's outcome once chat_stream's NDJSON loop ends normally. done_payload is
    Matricxon's final `done:true` line (empty if the generator closed first, e.g. a cancelled reply).
    """
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
    """Called from startup_service.run_shutdown_tasks. Hops to a worker thread rather than calling
    TracerProvider.shutdown on the event loop: same deadlock risk and fix as
    ollama_telemetry.shutdown_ollama_telemetry.
    """
    if _tracer_provider is not None:
        await asyncio.to_thread(_tracer_provider.shutdown)
