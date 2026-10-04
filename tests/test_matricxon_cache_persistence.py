"""Settings > Matricxon saved chat caches: the service fans out to every host through a mocked transport
(no real Matricxon), and the router maps a service failure onto a 502."""

import json

import httpx
import pytest
from fastapi import HTTPException

from app.routers import matricxon_cache_admin
from app.schemas import MatricxonCachePersistenceUpdate
from app.services import engine_service, matricxon_cache_persistence
from app.services.matricxon_cache_persistence import MatricxonCachePersistenceService
from app.services.matricxon_client import MatricxonError

A, B = "http://a:8420", "http://b:8420"


def _state(files: int = 1, used: int = 100, enabled: bool = True) -> dict:
    return {"enabled": enabled, "budget_mb": 4096, "ttl_hours": 168, "files": files, "used_bytes": used}


def _service(handler, hosts: list[str]) -> tuple[MatricxonCachePersistenceService, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def recording(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    return MatricxonCachePersistenceService(hosts, httpx.MockTransport(recording)), seen


@pytest.mark.asyncio
async def test_status_sums_files_and_bytes_across_hosts() -> None:
    states = {A: _state(2, 300), B: _state(1, 50, enabled=False)}
    service, _ = _service(
        lambda r: httpx.Response(200, json=states[f"{r.url.scheme}://{r.url.host}:{r.url.port}"]), [A, B]
    )
    result = await service.status()
    assert (result.files, result.used_bytes, result.hosts) == (3, 350, 2)
    assert result.enabled is False  # only on if every host has it on
    assert (result.budget_mb, result.ttl_hours, result.unreachable) == (4096, 168, [])


@pytest.mark.asyncio
async def test_update_sends_only_the_changed_fields_to_every_host() -> None:
    service, seen = _service(lambda r: httpx.Response(200, json=_state()), [A, B])
    await service.update(MatricxonCachePersistenceUpdate(budget_mb=10))
    assert [r.method for r in seen] == ["PUT", "PUT"]
    assert all(json.loads(r.content) == {"budget_mb": 10} for r in seen)
    assert all(r.url.path == "/api/cache/persistence" for r in seen)


@pytest.mark.asyncio
async def test_clear_uses_delete() -> None:
    service, seen = _service(lambda r: httpx.Response(200, json=_state(0, 0)), [A])
    result = await service.clear()
    assert seen[0].method == "DELETE" and result.files == 0


@pytest.mark.asyncio
async def test_one_unreachable_host_is_reported_but_the_rest_still_answer() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "b":
            raise httpx.ConnectError("down")
        return httpx.Response(200, json=_state())

    service, _ = _service(handler, [A, B])
    result = await service.status()
    assert (result.hosts, result.unreachable) == (1, [B])


@pytest.mark.asyncio
async def test_an_old_matricxon_without_the_endpoint_gets_a_clear_message() -> None:
    service, _ = _service(lambda r: httpx.Response(404), [A])
    with pytest.raises(MatricxonError, match="no disk cache"):
        await service.status()


@pytest.mark.asyncio
async def test_no_hosts_at_all_is_an_error() -> None:
    service, _ = _service(lambda r: httpx.Response(200, json=_state()), [])
    with pytest.raises(MatricxonError):
        await service.status()


@pytest.mark.asyncio
async def test_router_turns_a_service_failure_into_a_502(db, monkeypatch: pytest.MonkeyPatch) -> None:
    class Failing(MatricxonCachePersistenceService):
        async def status(self):
            raise MatricxonError("boom")

    await engine_service.set_active_engine(db, "matricxon")
    monkeypatch.setattr(matricxon_cache_admin, "MatricxonCachePersistenceService", Failing)
    with pytest.raises(HTTPException) as caught:
        await matricxon_cache_admin.get_cache_persistence(db=db, _admin=None)
    assert caught.value.status_code == 502 and caught.value.detail == "boom"


@pytest.mark.asyncio
async def test_router_passes_the_update_through(db, monkeypatch: pytest.MonkeyPatch) -> None:
    await engine_service.set_active_engine(db, "matricxon")
    service, seen = _service(lambda r: httpx.Response(200, json=_state()), [A])
    monkeypatch.setattr(matricxon_cache_persistence.matricxon_pool, "get_effective_hosts", lambda: [A])
    monkeypatch.setattr(matricxon_cache_admin, "MatricxonCachePersistenceService", lambda: service)
    result = await matricxon_cache_admin.update_cache_persistence(
        MatricxonCachePersistenceUpdate(enabled=False), db=db, _admin=None
    )
    assert result.files == 1 and json.loads(seen[0].content) == {"enabled": False}


@pytest.mark.asyncio
@pytest.mark.parametrize("call", ["get", "put", "delete"])
async def test_every_endpoint_refuses_while_ollama_is_the_active_engine(db, call: str) -> None:
    await engine_service.set_active_engine(db, "ollama")
    calls = {
        "get": lambda: matricxon_cache_admin.get_cache_persistence(db=db, _admin=None),
        "put": lambda: matricxon_cache_admin.update_cache_persistence(
            MatricxonCachePersistenceUpdate(enabled=True), db=db, _admin=None
        ),
        "delete": lambda: matricxon_cache_admin.clear_cache_persistence(db=db, _admin=None),
    }
    with pytest.raises(HTTPException) as caught:
        await calls[call]()
    assert caught.value.status_code == 409 and "Matricxon" in caught.value.detail


@pytest.mark.asyncio
async def test_forget_chat_deletes_on_every_host_and_never_raises() -> None:
    service, seen = _service(lambda r: httpx.Response(200, json=_state()), [A, B])
    await service.forget_chat("conv-1")
    assert [(r.method, r.url.path) for r in seen] == [("DELETE", "/api/cache/persistence/chats/conv-1")] * 2

    broken, _ = _service(lambda r: httpx.Response(500), [A])
    await broken.forget_chat("conv-1")  # an unreachable/old host only gets logged
