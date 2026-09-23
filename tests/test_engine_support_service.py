"""Unit tests for app/services/engine_support_service.py and its GET /api/stats/engine-support router — both
network-touching dependencies (engine_service.current_engine, matricxon_client.get_capabilities) are
monkeypatched on this service module's own imported bindings (see the module-split import-binding gotcha)."""

import pytest

from app.routers import stats
from app.services import engine_support_service as svc
from app.services.engine_support_service import EngineSupportService
from app.services.matricxon_client import MatricxonError


def _set_engine(monkeypatch, engine: str) -> None:
    monkeypatch.setattr(svc.engine_service, "current_engine", lambda: engine)


@pytest.mark.asyncio
async def test_ollama_is_reported_unavailable_without_asking_matricxon(monkeypatch):
    _set_engine(monkeypatch, "ollama")

    async def fail():
        raise AssertionError("Matricxon must not be queried while Ollama is active")

    monkeypatch.setattr(svc.matricxon_client, "get_capabilities", fail)

    result = await EngineSupportService.get()

    assert result.active_engine == "ollama"
    assert result.available is False
    assert result.architectures == [] and result.quantizations == []


@pytest.mark.asyncio
async def test_matricxon_returns_its_real_capability_lists(monkeypatch):
    _set_engine(monkeypatch, "matricxon")

    async def capabilities():
        return {"supported_architectures": ["llama", "phi2"], "supported_quantizations": ["F16", "Q4_K"]}

    monkeypatch.setattr(svc.matricxon_client, "get_capabilities", capabilities)

    result = await EngineSupportService.get()

    assert result.available is True
    assert result.architectures == ["llama", "phi2"]
    assert result.quantizations == ["F16", "Q4_K"]
    assert result.error is None


@pytest.mark.asyncio
async def test_unreachable_matricxon_reports_an_error_instead_of_raising(monkeypatch):
    _set_engine(monkeypatch, "matricxon")

    async def unreachable():
        raise MatricxonError("Could not reach Matricxon")

    monkeypatch.setattr(svc.matricxon_client, "get_capabilities", unreachable)

    result = await EngineSupportService.get()

    assert result.available is True
    assert result.error == "Could not reach Matricxon"
    assert result.architectures == []


@pytest.mark.asyncio
async def test_router_returns_the_service_result(monkeypatch, admin_user):
    _set_engine(monkeypatch, "ollama")

    result = await stats.get_engine_support(_user=admin_user)

    assert result.available is False
