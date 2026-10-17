"""Settings > Matricxon: the encrypted on-disk prompt cache (../matricxon/app/runtime/cache_store.py). Its
limits live in Matricxon and change instantly (GET/PUT/DELETE /api/cache/persistence), so nothing is saved
here or restarted. With several hosts each is asked and results combined; each keeps its own cache on its
own disk (unless the folder is shared, as in a cluster: then files are counted once).
"""

import logging

import httpx

from app.schemas import MatricxonCachePersistence, MatricxonCachePersistenceUpdate
from app.services import matricxon_http, matricxon_pool
from app.services.matricxon_client import MatricxonError

logger = logging.getLogger("llama_chat")
_PATH = "/api/cache/persistence"
_TIMEOUT = httpx.Timeout(30.0, connect=5.0)


class MatricxonCachePersistenceService:
    def __init__(
        self,
        hosts: list[str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        shared_folder: bool = False,
    ) -> None:
        self._hosts = hosts if hosts is not None else matricxon_pool.get_effective_hosts()
        self._transport = transport
        # In a cluster every node reads the same saved-caches folder and reports the same files: count them once, not
        # per node.
        self._shared = shared_folder

    async def status(self) -> MatricxonCachePersistence:
        return await self._call_all("GET")

    async def update(self, change: MatricxonCachePersistenceUpdate) -> MatricxonCachePersistence:
        return await self._call_all("PUT", change.model_dump(exclude_none=True))

    async def clear(self) -> MatricxonCachePersistence:
        return await self._call_all("DELETE")

    async def forget_chat(self, conversation_id: str) -> None:
        """Best effort: drops a deleted chat's stored prompt caches on every host (ChatRequest.cache_tag)."""
        try:
            await self._call_all("DELETE", path=f"{_PATH}/chats/{conversation_id}")
        except MatricxonError as exc:
            logger.info("could not drop the cached prompts of a deleted chat: %s", exc)

    async def _call_all(self, method: str, body: dict | None = None, path: str = _PATH) -> MatricxonCachePersistence:
        results: list[dict] = []
        unreachable: list[str] = []
        errors: list[str] = []
        async with matricxon_http.client(timeout=_TIMEOUT, transport=self._transport) as client:
            for host in self._hosts:
                try:
                    resp = await client.request(method, host + path, json=body)
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
        files, chats, used = self._totals(results)
        return MatricxonCachePersistence(
            enabled=all(r["enabled"] for r in results),
            budget_mb=results[0]["budget_mb"],
            ttl_hours=results[0]["ttl_hours"],
            files=files,
            chats=chats,
            used_bytes=used,
            hosts=len(results),
            unreachable=unreachable,
        )

    def _totals(self, results: list[dict]) -> tuple[int, int, int]:
        """(files, chats, bytes); a Matricxon without `chats` counts its files. Nodes reading the same shared
        folder report the same files: count each folder once. Forced
        for a cluster; otherwise a node says if it is on a network mount and where (older ones say nothing: own disk).
        """
        seen: set[str] = set()
        files = chats = used = 0
        for r in results:
            folder = r.get("folder") if r.get("shared") else None
            if self._shared:
                folder = "cluster"
            if folder is not None:
                if folder in seen:
                    continue
                seen.add(folder)
            files += r["files"]
            chats += r.get("chats", r["files"])
            used += r["used_bytes"]
        return files, chats, used
