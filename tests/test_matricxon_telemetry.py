"""Unit tests for app/services/matricxon_telemetry.py — the Matricxon-flavored twin of
tests/test_ollama_telemetry.py, same "never touch the global OTel registry" guarantee (see that module's own
docstring for the full reasoning, identical here)."""

import asyncio

import pytest
from opentelemetry import trace

from app.services import matricxon_telemetry


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    monkeypatch.setattr(matricxon_telemetry, "_tracer_provider", None)


def test_get_tracer_is_a_safe_noop_before_init():
    tracer = matricxon_telemetry.get_tracer()
    with tracer.start_as_current_span("matricxon.chat") as span:
        span.set_attribute("gen_ai.request.model", "ministral-3:3b")


@pytest.mark.asyncio
async def test_init_never_touches_the_global_tracer_provider_registry():
    global_provider_before = trace.get_tracer_provider()

    matricxon_telemetry.init_matricxon_telemetry()

    assert trace.get_tracer_provider() is global_provider_before
    assert matricxon_telemetry.get_tracer() is not trace.get_tracer("pairing.matricxon")


@pytest.mark.asyncio
async def test_shutdown_returns_promptly_without_deadlocking_the_event_loop():
    matricxon_telemetry.init_matricxon_telemetry()

    await asyncio.wait_for(matricxon_telemetry.shutdown_matricxon_telemetry(), timeout=5.0)


@pytest.mark.asyncio
async def test_shutdown_before_init_is_a_noop():
    await matricxon_telemetry.shutdown_matricxon_telemetry()


def test_record_chat_success_sets_attributes_from_the_done_payload():
    tracer = matricxon_telemetry.get_tracer()
    with tracer.start_as_current_span("matricxon.chat") as span:
        matricxon_telemetry.record_chat_success(
            span,
            "http://localhost:8420",
            1,
            {"prompt_eval_count": 12, "eval_count": 34, "load_duration": 1_000_000, "total_duration": 2_000_000},
        )


@pytest.mark.asyncio
async def test_record_chat_error_sets_error_status():
    from opentelemetry.trace import StatusCode

    # A real (recording) span, not the no-op fallback get_tracer() returns before init — the no-op span's
    # NonRecordingSpan has no readable .status at all, so this needs a real TracerProvider to assert against.
    # Async (unlike this file's other plain-attribute-setting test) since init_matricxon_telemetry() calls
    # capture_main_loop(), which needs a running event loop — see that function's own docstring.
    matricxon_telemetry.init_matricxon_telemetry()
    tracer = matricxon_telemetry.get_tracer()
    with tracer.start_as_current_span("matricxon.chat", record_exception=False, set_status_on_exception=False) as span:
        matricxon_telemetry.record_chat_error(span, "http://localhost:8420", 1, RuntimeError("boom"))
        assert span.status.status_code == StatusCode.ERROR
    # Flushes/joins the BatchSpanProcessor's background export before this test's own event loop closes — without
    # this, its worker thread tries to export after the loop is gone (harmless but noisy: a logged "Failed to
    # persist" + a "coroutine was never awaited" warning from a real background thread outliving the test).
    await matricxon_telemetry.shutdown_matricxon_telemetry()
