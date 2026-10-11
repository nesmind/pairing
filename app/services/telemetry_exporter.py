"""The cross-thread bridge of the Telemetry page's OpenTelemetry export, separate from SDK bootstrap
(ollama_telemetry/matricxon_telemetry both use this DBSpanExporter unmodified; their spans arrive under scope
names "pairing.ollama"/"pairing.matricxon", which _persist_spans uses to stamp TelemetryEvent.engine), so the
background-thread-to-async-DB hazard is testable alone.

opentelemetry-sdk's BatchSpanProcessor calls a SpanExporter's export from its own worker thread, not the
asyncio loop, but persisting a span is an async SQLAlchemy write that must run *on* the loop.
capture_main_loop stashes it (called once from ollama_telemetry.init_ollama_telemetry, inside
startup_service.run_startup_tasks on the main loop); DBSpanExporter.export hands each batch off via
asyncio.run_coroutine_threadsafe(...).result(timeout=...), bounded so a stuck DB can't wedge the SDK's worker.
Any failure (loop not captured, timeout, DB error) is logged and returned as SpanExportResult.FAILURE, never
raised: telemetry must not affect the Ollama/Matricxon call it describes.
"""

import asyncio
import logging
from collections.abc import Sequence
from datetime import UTC, datetime

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode

from app.database import AsyncSessionLocal
from app.models import TelemetryEvent

logger = logging.getLogger("llama_chat")

_EXPORT_TIMEOUT_SECONDS = 10.0

_main_loop: asyncio.AbstractEventLoop | None = None


def capture_main_loop() -> None:
    """Call once from inside the app's running event loop (ollama_telemetry.init_ollama_telemetry): a background thread
    has no other way back to it."""
    global _main_loop
    _main_loop = asyncio.get_running_loop()


def _ns_to_ms(value: object) -> float | None:
    return value / 1_000_000 if isinstance(value, int | float) else None


def _span_to_event(span: ReadableSpan, engine: str) -> TelemetryEvent | None:
    """Converts one finished span to a row; None for a span lacking the timestamps a row needs (defensive: this
    app's spans always have both).
    """
    if span.start_time is None or span.end_time is None:
        return None
    attrs = span.attributes or {}
    return TelemetryEvent(
        engine=engine,
        # Span names are "<engine>.chat"/"<engine>.embed" (the start_as_current_span calls in ollama_client.py/
        # matricxon_client.py). Splitting on the first "." rather than removeprefix(f"{engine}.") reads the same today
        # but yields no wrong kind if a span's prefix drifts from `engine`.
        kind=span.name.split(".", 1)[-1],
        model=attrs.get("gen_ai.request.model"),
        host=attrs.get("server.address", "unknown"),
        success=span.status.status_code != StatusCode.ERROR,
        error_message=span.status.description,
        started_at=datetime.fromtimestamp(span.start_time / 1_000_000_000, tz=UTC),
        duration_ms=(span.end_time - span.start_time) / 1_000_000,
        total_duration_ms=_ns_to_ms(attrs.get("ollama.total_duration_ns")),
        load_duration_ms=_ns_to_ms(attrs.get("ollama.load_duration_ns")),
        prompt_eval_duration_ms=_ns_to_ms(attrs.get("ollama.prompt_eval_duration_ns")),
        eval_duration_ms=_ns_to_ms(attrs.get("ollama.eval_duration_ns")),
        prompt_eval_count=attrs.get("gen_ai.usage.input_tokens"),
        eval_count=attrs.get("gen_ai.usage.output_tokens"),
        attempt_count=attrs.get("ollama.attempt_count", 1),
    )


# Every tracer comes from one of these two scopes (ollama_telemetry.get_tracer / matricxon_telemetry.get_tracer);
# a span from any other (e.g. a stray no-op-tracer span from a test that bootstrapped neither) is dropped, not guessed.
_ENGINE_BY_SCOPE = {"pairing.ollama": "ollama", "pairing.matricxon": "matricxon"}


async def _persist_spans(spans: Sequence[ReadableSpan]) -> None:
    async with AsyncSessionLocal() as db:
        events = []
        for span in spans:
            scope_name = span.instrumentation_scope.name if span.instrumentation_scope else None
            engine = _ENGINE_BY_SCOPE.get(scope_name)
            if engine is None:
                continue
            event = _span_to_event(span, engine)
            if event is not None:
                events.append(event)
        if events:
            db.add_all(events)
            await db.commit()


class DBSpanExporter(SpanExporter):
    """Registered on a BatchSpanProcessor (never SimpleSpanProcessor: that calls export synchronously on the
    engine call's hot path, the coupling this file avoids).
    """

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        if _main_loop is None:
            logger.warning("Dropping %d telemetry span(s): main event loop not captured yet.", len(spans))
            return SpanExportResult.FAILURE
        try:
            future = asyncio.run_coroutine_threadsafe(_persist_spans(spans), _main_loop)
            future.result(timeout=_EXPORT_TIMEOUT_SECONDS)
        except Exception:
            logger.exception("Failed to persist %d telemetry span(s).", len(spans))
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True
