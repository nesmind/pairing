"""The cross-thread bridge behind the Telemetry page's OpenTelemetry
export pipeline — kept in its own file, independent of SDK bootstrap
(see app.services.ollama_telemetry), so the actual hazard here (moving
data from a background thread into this app's async DB layer) is
testable on its own.

opentelemetry-sdk's BatchSpanProcessor calls a SpanExporter's export()
from its own dedicated worker thread, not this app's asyncio event
loop — but persisting a span means an async SQLAlchemy write, which
needs to run *on* that loop. capture_main_loop() stashes a reference to
it (called once, from ollama_telemetry.init_ollama_telemetry(), which
itself runs inside startup_service.run_startup_tasks() on the main
loop); DBSpanExporter.export() then hands each batch off via
asyncio.run_coroutine_threadsafe(...).result(timeout=...) — bounded, so
a stuck DB never wedges the SDK's worker thread forever. Any failure
(loop not captured yet, timeout, DB error) is logged and turned into
SpanExportResult.FAILURE — never raised, since a telemetry hiccup must
never be allowed to affect the real Ollama call it's describing.
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
    """Must be called once, from inside the app's own running event
    loop (see ollama_telemetry.init_ollama_telemetry) — there's no other
    reliable way for a plain background thread to find its way back to
    it later."""
    global _main_loop
    _main_loop = asyncio.get_running_loop()


def _ns_to_ms(value: object) -> float | None:
    return value / 1_000_000 if isinstance(value, int | float) else None


def _span_to_event(span: ReadableSpan) -> TelemetryEvent | None:
    """Converts one finished span into a row — returns None for a span
    missing the timestamps a row can't do without (defensive only; every
    span this app itself creates always has both)."""
    if span.start_time is None or span.end_time is None:
        return None
    attrs = span.attributes or {}
    return TelemetryEvent(
        kind=span.name.removeprefix("ollama."),
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


async def _persist_spans(spans: Sequence[ReadableSpan]) -> None:
    async with AsyncSessionLocal() as db:
        events = []
        for span in spans:
            if span.instrumentation_scope is None or span.instrumentation_scope.name != "pairing.ollama":
                continue
            event = _span_to_event(span)
            if event is not None:
                events.append(event)
        if events:
            db.add_all(events)
            await db.commit()


class DBSpanExporter(SpanExporter):
    """Registered on a BatchSpanProcessor (never SimpleSpanProcessor —
    that would call export() synchronously inline on the Ollama call's
    own hot path, exactly the coupling this file exists to avoid)."""

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
