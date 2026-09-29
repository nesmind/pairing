"""Unit tests for app/services/engine_support_service.py and its GET /api/stats/engine-support router. The
active engine is set via engine_service._cached_engine directly (same pattern tests/test_health.py already
uses); matricxon_client.get_capabilities (reached through MatricxonEngine.get_format_support — see
tests/test_engine_adapters.py for that forwarding link itself) is monkeypatched on its own module, not on any
particular importer's binding of it, since module objects are singletons regardless of which file imports them
(see the module-split import-binding gotcha)."""

import pytest

from app.routers import stats
from app.services import engine_service, matricxon_client
from app.services.engine_support_service import EngineSupportService
from app.services.matricxon_client import MatricxonError


@pytest.fixture(autouse=True)
def _reset_engine_cache():
    engine_service._cached_engine = engine_service.DEFAULT_ENGINE
    yield
    engine_service._cached_engine = engine_service.DEFAULT_ENGINE


@pytest.mark.asyncio
async def test_ollama_is_reported_unavailable_without_asking_matricxon(monkeypatch):
    engine_service._cached_engine = "ollama"

    async def fail():
        raise AssertionError("Matricxon must not be queried while Ollama is active")

    monkeypatch.setattr(matricxon_client, "get_capabilities", fail)

    result = await EngineSupportService.get()

    assert result.active_engine == "ollama"
    assert result.available is False
    assert result.architectures == [] and result.quantizations == []
    assert result.moe_architectures == []


@pytest.mark.asyncio
async def test_matricxon_returns_its_real_capability_lists(monkeypatch):
    engine_service._cached_engine = "matricxon"

    async def capabilities():
        return {
            "supported_architectures": ["llama", "phi2"],
            "supported_quantizations": ["F16", "Q4_K"],
            "moe_supported_architectures": ["llama"],
        }

    monkeypatch.setattr(matricxon_client, "get_capabilities", capabilities)

    result = await EngineSupportService.get()

    assert result.available is True
    assert result.architectures == ["llama", "phi2"]
    assert result.quantizations == ["F16", "Q4_K"]
    assert result.moe_architectures == ["llama"]
    assert result.error is None


@pytest.mark.asyncio
async def test_unreachable_matricxon_reports_an_error_instead_of_raising(monkeypatch):
    engine_service._cached_engine = "matricxon"

    async def unreachable():
        raise MatricxonError("Could not reach Matricxon")

    monkeypatch.setattr(matricxon_client, "get_capabilities", unreachable)

    result = await EngineSupportService.get()

    assert result.available is True
    assert result.error == "Could not reach Matricxon"
    assert result.architectures == []


@pytest.mark.asyncio
async def test_router_returns_the_service_result(monkeypatch, admin_user):
    engine_service._cached_engine = "ollama"

    result = await stats.get_engine_support(_user=admin_user)

    assert result.available is False
