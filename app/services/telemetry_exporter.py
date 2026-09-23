"""The cross-thread bridge behind the Telemetry page's OpenTelemetry
export pipeline — kept in its own file, independent of SDK bootstrap
(see app.services.ollama_telemetry/matricxon_telemetry, both of which use this same DBSpanExporter unmodified —
each engine's spans arrive under their own instrumentation scope name, "pairing.ollama"/"pairing.matricxon",
which is how _persist_spans below tells them apart and stamps the right TelemetryEvent.engine), so the actual
hazard here (moving data from a background thread into this app's async DB layer) is testable on its own.

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
never be allowed to affect the real Ollama/Matricxon call it's describing.
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


def _span_to_event(span: ReadableSpan, engine: str) -> TelemetryEvent | None:
    """Converts one finished span into a row — returns None for a span
    missing the timestamps a row can't do without (defensive only; every
    span this app itself creates always has both)."""
    if span.start_time is None or span.end_time is None:
        return None
    attrs = span.attributes or {}
    return TelemetryEvent(
        engine=engine,
        # Span names are "<engine>.chat"/"<engine>.embed" (see ollama_client.py/matricxon_client.py's own
        # start_as_current_span calls) — splitting on the first "." rather than removeprefix(f"{engine}.") reads
        # the same either way today, but doesn't silently produce a wrong kind if a span's own name prefix ever
        # drifted from `engine` for some reason.
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


# Every tracer this app creates comes from one of these two scopes (see ollama_telemetry.get_tracer/
# matricxon_telemetry.get_tracer) — a span from anything else (there is nothing else today, but a stray
# no-op-tracer span from a test that never bootstrapped either provider is a real, harmless possibility) is
# silently dropped rather than guessed at.
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
    """Registered on a BatchSpanProcessor (never SimpleSpanProcessor —
    that would call export() synchronously inline on the ML engine call's
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
