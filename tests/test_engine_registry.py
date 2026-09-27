"""Unit tests for app/services/engines/registry.py — duplicate/unknown-name handling, and the parity check that
stops EngineName (the Pydantic Literal request validation needs) from drifting from the registry's own real
engine set the way app.services.engine_service.DEFAULT_ENGINE and schemas.common.ActiveEngineConfig's default
already had (see that fix)."""

from typing import get_args

import pytest

from app.schemas import EngineName
from app.services.engines.base import EngineCapabilities, EngineCapabilityError, InferenceEngine
from app.services.engines.registry import EngineRegistry, registry


class _FakeEngine(InferenceEngine):
    def __init__(self, name: str) -> None:
        self.name = name
        self.display_name = name
        self.capabilities = EngineCapabilities(
            embeddings=False,
            model_management=False,
            stop_model=False,
            local_process=False,
            host_pool=False,
            support_checking=False,
            format_introspection=False,
        )
        self.error_types = ()

    async def list_models(self):
        return []

    def chat_stream(self, model, messages, params):
        async def _gen():
            return
            yield

        return _gen()

    async def chat_once(self, model, messages, params=None):
        return ""

    async def check_health(self):
        return {}


def test_engine_name_literal_matches_the_real_registry():
    assert set(get_args(EngineName)) == set(registry.names())


def test_get_returns_the_matching_engine():
    fake = _FakeEngine("fake")
    fake_registry = EngineRegistry([fake])
    assert fake_registry.get("fake") is fake


def test_get_raises_a_clear_error_for_an_unknown_name():
    fake_registry = EngineRegistry([_FakeEngine("fake")])
    with pytest.raises(KeyError, match="fake"):
        fake_registry.get("nonexistent")


def test_duplicate_engine_names_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        EngineRegistry([_FakeEngine("dup"), _FakeEngine("dup")])


def test_all_error_types_includes_engine_capability_error_and_every_engines_own():
    fake_a = _FakeEngine("a")
    fake_a.error_types = (RuntimeError,)
    fake_b = _FakeEngine("b")
    fake_b.error_types = (ValueError,)
    fake_registry = EngineRegistry([fake_a, fake_b])

    types = fake_registry.all_error_types()

    assert EngineCapabilityError in types
    assert RuntimeError in types
    assert ValueError in types
