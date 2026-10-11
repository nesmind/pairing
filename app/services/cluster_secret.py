"""Sets the admin secret on every Matricxon node, then in pAIring.

Nodes check pAIring's signed admin calls against it (../matricxon/app/server/admin_auth.py). Changing it
needs every node to answer, so none is left on the old secret; a failure part-way puts the old one back.
"""

import logging

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.services import matricxon_http, matricxon_pool, server_pool_broadcast, settings_service
from app.services.cluster_errors import NodeNotAllowedError

logger = logging.getLogger("llama_chat")
_TIMEOUT = httpx.Timeout(20.0, connect=5.0)


class ClusterSecret:
    def __init__(self, db: AsyncSession, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._db = db
        self._transport = transport

    async def set(self, secret: str) -> None:
        """ "" removes the secret (the nodes' admin routes become open)."""
        config = await settings_service.get_matricxon_server_config(self._db)
        hosts = matricxon_pool.get_effective_hosts()
        old = config.admin_secret or ""
        done: list[str] = []
        async with matricxon_http.client(timeout=_TIMEOUT, transport=self._transport) as client:
            silent = [h for h in hosts if not await self._answers(client, h)]
            if silent:
                raise NodeNotAllowedError(
                    "Every node must answer to change the secret; not answering: " + ", ".join(silent)
                )
            for host in hosts:
                problem = await self._put(client, host, secret)
                if problem:
                    await self._roll_back(done, old, secret)
                    raise NodeNotAllowedError(f"{host} did not accept the secret ({problem}); nothing changed.")
                done.append(host)
        config.admin_secret = secret or None
        await settings_service.set_matricxon_server_config(self._db, config)
        matricxon_pool.refresh_from_config(config)
        await server_pool_broadcast.broadcast_refresh("/api/settings/matricxon/internal-refresh")
        logger.info("cluster: admin secret %s on %d node(s)", "set" if secret else "removed", len(hosts))

    @staticmethod
    async def _answers(client: httpx.AsyncClient, host: str) -> bool:
        try:
            return (await client.get(host + "/api/cluster/readiness")).status_code == 200
        except httpx.HTTPError:
            return False

    @staticmethod
    async def _put(client: httpx.AsyncClient, host: str, secret: str) -> str | None:
        try:
            resp = await client.put(host + "/api/cluster/admin-secret", json={"secret": secret})
        except httpx.HTTPError as exc:
            return str(exc) or type(exc).__name__
        if resp.status_code == 404:
            return "update its Matricxon"
        return None if resp.is_success else f"HTTP {resp.status_code}"

    async def _roll_back(self, done: list[str], old: str, new: str) -> None:
        """Nodes that took the new secret get the old one back (signed with the new one)."""
        matricxon_http.set_secret(new or None)
        try:
            async with matricxon_http.client(timeout=_TIMEOUT, transport=self._transport) as client:
                for host in done:
                    await self._put(client, host, old)
        finally:
            matricxon_http.set_secret(old or None)
