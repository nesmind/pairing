"""Unit tests for app/services/ollama_telemetry.py — mainly the
"never touch the global OTel registry" guarantee (see that module's own
docstring for why: a test that called opentelemetry.trace.set_tracer_provider
would leak into every other test in the same pytest process, since
that's process-global, effectively set-once state)."""

import asyncio

import pytest
from opentelemetry import trace

from app.services import ollama_telemetry


@pytest.fixture(autouse=True)
def _reset_module_state(monkeypatch):
    monkeypatch.setattr(ollama_telemetry, "_tracer_provider", None)


def test_get_tracer_is_a_safe_noop_before_init():
    tracer = ollama_telemetry.get_tracer()
    with tracer.start_as_current_span("ollama.chat") as span:
        span.set_attribute("gen_ai.request.model", "llama3:latest")


@pytest.mark.asyncio
async def test_init_never_touches_the_global_tracer_provider_registry():
    global_provider_before = trace.get_tracer_provider()

    ollama_telemetry.init_ollama_telemetry()

    assert trace.get_tracer_provider() is global_provider_before
    assert ollama_telemetry.get_tracer() is not trace.get_tracer("pairing.ollama")


@pytest.mark.asyncio
async def test_shutdown_returns_promptly_without_deadlocking_the_event_loop():
    ollama_telemetry.init_ollama_telemetry()

    await asyncio.wait_for(ollama_telemetry.shutdown_ollama_telemetry(), timeout=5.0)


@pytest.mark.asyncio
async def test_shutdown_before_init_is_a_noop():
    await ollama_telemetry.shutdown_ollama_telemetry()
