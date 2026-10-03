"""Settings > Matricxon: the encrypted on-disk prompt cache (see ../matricxon/app/runtime/cache_store.py).
Its limits live in Matricxon itself and change there instantly (GET/PUT/DELETE /api/cache/persistence), so
there is nothing to save in this app's own database or to restart for. With several hosts configured every
one is asked and the results combined — each host keeps its own cache on its own disk."""

import httpx

from app.schemas import MatricxonCachePersistence, MatricxonCachePersistenceUpdate
from app.services import matricxon_pool
from app.services.matricxon_client import MatricxonError

_PATH = "/api/cache/persistence"
_TIMEOUT = httpx.Timeout(30.0, connect=5.0)


class MatricxonCachePersistenceService:
    def __init__(self, hosts: list[str] | None = None, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._hosts = hosts if hosts is not None else matricxon_pool.get_effective_hosts()
        self._transport = transport

    async def status(self) -> MatricxonCachePersistence:
        return await self._call_all("GET")

    async def update(self, change: MatricxonCachePersistenceUpdate) -> MatricxonCachePersistence:
        return await self._call_all("PUT", change.model_dump(exclude_none=True))

    async def clear(self) -> MatricxonCachePersistence:
        return await self._call_all("DELETE")

    async def _call_all(self, method: str, body: dict | None = None) -> MatricxonCachePersistence:
        results: list[dict] = []
        unreachable: list[str] = []
        errors: list[str] = []
        async with httpx.AsyncClient(timeout=_TIMEOUT, transport=self._transport) as client:
            for host in self._hosts:
                try:
                    resp = await client.request(method, host + _PATH, json=body)
                except httpx.HTTPError as exc:
                    unreachable.append(host)
                    errors.append(f"{host}: {exc}")
                    continue
                if resp.status_code == 404:
                    unreachable.append(host)
                    errors.append(f"{host}: this Matricxon version has no disk cache — update it")
                elif resp.is_error:
                    unreachable.append(host)
                    errors.append(f"{host}: HTTP {resp.status_code}")
                else:
                    results.append(resp.json())
        if not results:
            raise MatricxonError("; ".join(errors) or "no Matricxon host is configured")
        return MatricxonCachePersistence(
            enabled=all(r["enabled"] for r in results),
            budget_mb=results[0]["budget_mb"],
            ttl_hours=results[0]["ttl_hours"],
            files=sum(r["files"] for r in results),
            used_bytes=sum(r["used_bytes"] for r in results),
            hosts=len(results),
            unreachable=unreachable,
        )
