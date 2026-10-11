"""Tests for services/telemetry_exporter.py, the cross-thread bridge of the Telemetry page's OpenTelemetry
export. Uses a real file-backed engine, not the shared `db`'s:memory: (see tests/test_reply_generation_service.py: a
genuinely concurrent writer, here a real background thread, needs its own connection, not the one shared:memory:
connection).
"""

import asyncio
import os
import tempfile

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import Status, StatusCode
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.database import Base
from app.models import TelemetryEvent
from app.services import telemetry_exporter
from app.services.telemetry_exporter import DBSpanExporter


async def _build_file_session_factory():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


def _make_readable_span(name="ollama.chat", scope="pairing.ollama", success=True, attributes=None):
    """Builds a real ReadableSpan via a throwaway TracerProvider + capturing exporter: the simplest genuine instance
    (real timestamps, instrumentation_scope) without a hand-rolled fake that could drift.
    """

    class _Capture(SpanExporter):
        def __init__(self):
            self.spans = []

        def export(self, spans):
            self.spans.extend(spans)
            return SpanExportResult.SUCCESS

        def shutdown(self):
            pass

    capture = _Capture()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(capture))
    tracer = provider.get_tracer(scope)
    with tracer.start_as_current_span(name, record_exception=False, set_status_on_exception=False) as span:
        for key, value in (attributes or {}).items():
            span.set_attribute(key, value)
        if not success:
            span.set_status(Status(StatusCode.ERROR, "boom"))
    return capture.spans[0]


@pytest.mark.asyncio
async def test_export_returns_failure_when_main_loop_not_captured(monkeypatch):
    monkeypatch.setattr(telemetry_exporter, "_main_loop", None)
    span = _make_readable_span()

    result = DBSpanExporter().export([span])

    assert result == SpanExportResult.FAILURE


@pytest.mark.asyncio
async def test_export_persists_spans_from_a_real_background_thread(monkeypatch):
    engine, session_factory = await _build_file_session_factory()
    monkeypatch.setattr(telemetry_exporter, "AsyncSessionLocal", session_factory)
    telemetry_exporter.capture_main_loop()

    span = _make_readable_span(
        attributes={
            "gen_ai.request.model": "llama3:latest",
            "server.address": "http://x:11434",
            "gen_ai.usage.input_tokens": 12,
            "gen_ai.usage.output_tokens": 34,
            "ollama.attempt_count": 1,
        }
    )

    # Mirrors production: BatchSpanProcessor calls export from its own worker thread, never the main loop's.
    result = await asyncio.to_thread(DBSpanExporter().export, [span])

    assert result == SpanExportResult.SUCCESS
    async with session_factory() as db:
        rows = (await db.execute(select(TelemetryEvent))).scalars().all()
    assert len(rows) == 1
    assert rows[0].kind == "chat"
    assert rows[0].engine == "ollama"
    assert rows[0].model == "llama3:latest"
    assert rows[0].host == "http://x:11434"
    assert rows[0].success is True
    assert rows[0].prompt_eval_count == 12
    assert rows[0].eval_count == 34
    await engine.dispose()
    os.remove(engine.url.database)


@pytest.mark.asyncio
async def test_export_persists_matricxon_spans_stamped_with_the_matricxon_engine(monkeypatch):
    """A "pairing.matricxon" scope (matricxon_telemetry.get_tracer) must produce a row with engine="matricxon", not be
    dropped as
    any scope other than "pairing.ollama" used to be before Matricxon
    (test_export_ignores_spans_from_a_different_instrumentation_scope
    now checks a real *third*, unrelated scope).
    """
    engine, session_factory = await _build_file_session_factory()
    monkeypatch.setattr(telemetry_exporter, "AsyncSessionLocal", session_factory)
    telemetry_exporter.capture_main_loop()

    span = _make_readable_span(
        name="matricxon.chat",
        scope="pairing.matricxon",
        attributes={"gen_ai.request.model": "ministral-3:3b", "server.address": "http://x:8420"},
    )

    result = await asyncio.to_thread(DBSpanExporter().export, [span])

    assert result == SpanExportResult.SUCCESS
    async with session_factory() as db:
        rows = (await db.execute(select(TelemetryEvent))).scalars().all()
    assert len(rows) == 1
    assert rows[0].engine == "matricxon"
    assert rows[0].kind == "chat"
    assert rows[0].model == "ministral-3:3b"
    await engine.dispose()
    os.remove(engine.url.database)


@pytest.mark.asyncio
async def test_export_ignores_spans_from_a_different_instrumentation_scope(monkeypatch):
    engine, session_factory = await _build_file_session_factory()
    monkeypatch.setattr(telemetry_exporter, "AsyncSessionLocal", session_factory)
    telemetry_exporter.capture_main_loop()

    span = _make_readable_span(scope="some.other.library")

    result = await asyncio.to_thread(DBSpanExporter().export, [span])

    assert result == SpanExportResult.SUCCESS
    async with session_factory() as db:
        rows = (await db.execute(select(TelemetryEvent))).scalars().all()
    assert rows == []
    await engine.dispose()
    os.remove(engine.url.database)


@pytest.mark.asyncio
async def test_export_survives_persist_failure(monkeypatch):
    telemetry_exporter.capture_main_loop()

    async def _boom(_spans):
        raise RuntimeError("db is down")

    monkeypatch.setattr(telemetry_exporter, "_persist_spans", _boom)
    span = _make_readable_span()

    result = await asyncio.to_thread(DBSpanExporter().export, [span])

    assert result == SpanExportResult.FAILURE


def test_error_span_is_recorded_as_unsuccessful(monkeypatch):
    span = _make_readable_span(success=False)
    event = telemetry_exporter._span_to_event(span, "ollama")

    assert event.success is False
    assert event.error_message == "boom"
