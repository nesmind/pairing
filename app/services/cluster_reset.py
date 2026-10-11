"""Reset cluster: every node back to fresh and cluster mode off; hosts stay as plain hosts.

Needs no quorum. Each node clears only its own setup and journal; shared folders are untouched.
Unreachable nodes are reported for a manual reset (Matricxon's scripts/reset_cluster_node.sh).
Nodes reset in parallel and are reported as they finish.
"""

import asyncio
import logging
from collections.abc import AsyncIterator

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.cluster import ClusterResetResult
from app.services import matricxon_http, matricxon_pool, matricxon_process, server_pool_broadcast, settings_service
from app.services.cluster_hosts import ClusterHosts

logger = logging.getLogger("llama_chat")
_TIMEOUT = httpx.Timeout(20.0, connect=5.0)


class ClusterReset:
    CONCURRENCY = 8

    def __init__(self, db: AsyncSession, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._db = db
        self._transport = transport

    async def run(self) -> ClusterResetResult:
        """The whole reset, reported at the end."""
        result = ClusterResetResult(reset=[], failed=[])
        async for event in self.stream():
            if event["event"] == "done":
                result = ClusterResetResult(reset=event["reset"], failed=event["failed"])
        return result

    async def stream(self) -> AsyncIterator[dict]:
        """Events: `start`, one `node` each as it finishes, then `done`."""
        config = await settings_service.get_matricxon_server_config(self._db)
        nodes = ClusterHosts.nodes(config)
        yield {"event": "start", "total": len(nodes)}
        done: list[str] = []
        failed: list[str] = []
        async with matricxon_http.client(timeout=_TIMEOUT, transport=self._transport) as client:
            gate = asyncio.Semaphore(self.CONCURRENCY)

            async def one(host: str) -> tuple[str, bool, str]:
                async with gate:
                    return (host, *await self._reset(client, host))

            for finished in asyncio.as_completed([one(h) for h in nodes]):
                host, ok, reason = await finished
                (done if ok else failed).append(host)
                yield {"event": "node", "host": host, "ok": ok, "reason": reason}
        logger.info("cluster: RESET - fresh: %s, not reset: %s", done, failed)
        config.cluster_role = "off"
        config.advertise_url = config.first_node = None
        config.models_path = config.prompt_cache_path = None
        if len(nodes) > 1:  # stay as plain remote hosts
            config.mode, config.remote_hosts = "remote", nodes
        config.cluster_hosts = []
        await settings_service.set_matricxon_server_config(self._db, config)
        matricxon_process.write_env_file(config)
        matricxon_pool.refresh_from_config(config)
        await server_pool_broadcast.broadcast_refresh("/api/settings/matricxon/internal-refresh")
        yield {"event": "done", "reset": done, "failed": failed}

    @staticmethod
    async def _reset(client: httpx.AsyncClient, host: str) -> tuple[bool, str]:
        """(reset?, why not); unanswering or older nodes need a manual reset."""
        try:
            resp = await client.post(host + "/api/cluster/reset", params={"force": "true"})
        except httpx.HTTPError as exc:
            return False, f"not reachable ({str(exc) or type(exc).__name__})"
        if resp.is_success:
            return True, ""
        if resp.status_code == 404:
            return False, "this Matricxon has no reset: update it"
        detail = resp.json().get("detail") if resp.content else None
        return False, f"refused: {detail}" if detail else f"HTTP {resp.status_code}"
