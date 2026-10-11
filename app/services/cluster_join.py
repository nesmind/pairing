"""The cluster id and joining an existing cluster: its address plus id; the rest is read from it."""

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.cluster import ClusterHost
from app.services import matricxon_http, matricxon_pool, server_pool_broadcast, settings_service
from app.services.cluster_hosts import ClusterHosts
from app.services.cluster_nodes import ClusterNodes, NodeNotAllowedError
from app.services.cluster_settings import ClusterSettings


class ClusterJoin:
    def __init__(self, db: AsyncSession, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._db = db
        self._transport = transport

    @staticmethod
    def id_of(hosts: list[ClusterHost]) -> str | None:
        """The id reported by nodes at the newest epoch."""
        seen = [h.consensus for h in hosts if h.consensus and h.consensus.get("cluster_id")]
        return max(seen, key=lambda c: c.get("epoch", 0))["cluster_id"] if seen else None

    async def join(self, address: str, cluster_id: str, admin_secret: str = "") -> None:
        """Points this pAIring at the cluster once the node reports that id."""
        config = await settings_service.get_matricxon_server_config(self._db)
        if config.cluster_role != "off":
            raise NodeNotAllowedError("Cluster mode is already on here: turn it off (or reset it) first.")
        host = ClusterNodes.normalize(address)
        status = await ClusterSettings(self._db, self._transport, [host]).status(host)
        if status is None:
            raise NodeNotAllowedError(f"{host} does not answer, or is not part of a cluster.")
        if not status.get("cluster_id"):
            raise NodeNotAllowedError(f"{host} has no cluster id yet (an older Matricxon, or no leader yet).")
        if status["cluster_id"].strip().lower() != cluster_id.strip().lower():
            raise NodeNotAllowedError(f"{host} belongs to another cluster.")
        await self._check_secret(host, status, admin_secret)
        paths = status.get("config", {}).get("cluster/paths") or {}
        config.mode, config.remote_hosts, config.cluster_hosts = "remote", [host], []
        config.cluster_role = "seed"
        config.admin_secret = admin_secret or None
        config.models_path = paths.get("models_dir") or None
        config.prompt_cache_path = paths.get("prompt_cache_dir") or None
        await settings_service.set_matricxon_server_config(self._db, config)
        matricxon_pool.refresh_from_config(config)
        await server_pool_broadcast.broadcast_refresh("/api/settings/matricxon/internal-refresh")

    async def _check_secret(self, host: str, status: dict, secret: str) -> None:
        """A cluster whose nodes have an admin secret needs the same one here: proved by a signed no-op."""
        if not status.get("admin_auth"):
            return
        if not secret:
            raise NodeNotAllowedError(f"{host} is protected: enter the cluster's admin secret.")
        matricxon_http.set_secret(secret)
        try:
            async with matricxon_http.client(timeout=5.0, transport=self._transport) as client:
                resp = await client.put(host + "/api/cluster/admin-secret", json={"secret": secret})
        except httpx.HTTPError as exc:
            raise NodeNotAllowedError(f"{host} could not be reached: {exc}") from exc
        finally:
            matricxon_http.set_secret(None)
        if resp.status_code == 401:
            raise NodeNotAllowedError("That is not the cluster's admin secret.")

    async def leave(self) -> None:
        """Cluster mode off; nodes stay listed as plain remote hosts and none is touched.
        The managed Matricxon becomes a remote host too.
        """
        config = await settings_service.get_matricxon_server_config(self._db)
        nodes = ClusterHosts.nodes(config)
        config.cluster_role = "off"
        config.mode, config.remote_hosts, config.cluster_hosts = "remote", nodes, []
        config.first_node = config.advertise_url = config.models_path = config.prompt_cache_path = None
        await settings_service.set_matricxon_server_config(self._db, config)
        matricxon_pool.refresh_from_config(config)
        await server_pool_broadcast.broadcast_refresh("/api/settings/matricxon/internal-refresh")
