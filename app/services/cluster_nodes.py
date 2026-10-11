"""Cluster page > nodes: check a candidate is a fresh Matricxon, push the cluster config (role,
shared folders) and add it to the pool. Matricxon restarts itself; nothing is moved
(see../matricxon/docs/CLUSTER_STORAGE.md section 7).
"""

import logging
from urllib.parse import urlparse

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.cluster import NodeCheck
from app.services import matricxon_http, matricxon_pool, server_pool_broadcast, settings_service
from app.services.cluster_errors import NodeNotAllowedError, NodeNotFreshError, NodeResetRefusedError
from app.services.cluster_hosts import ClusterHosts
from app.services.cluster_reach import ClusterReach
from app.services.cluster_settings import ClusterSettings
from app.services.cluster_voter_changes import VoterChanges

logger = logging.getLogger("llama_chat")
_TIMEOUT = httpx.Timeout(20.0, connect=5.0)
_LOOPBACK = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


class ClusterNodes(VoterChanges):
    def __init__(self, db: AsyncSession, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._db = db
        self._transport = transport

    @staticmethod
    def normalize(address: str) -> str:
        text = address.strip().rstrip("/")
        if "://" not in text:
            text = f"http://{text}"
        parsed = urlparse(text)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise NodeNotAllowedError(f"Not a valid node address: {address}")
        return f"{parsed.scheme}://{parsed.netloc}"

    @staticmethod
    def pool_key(address: str) -> tuple[str, str, int | None]:
        """Scheme, machine and port: localhost, 127.0.0.1 and:1 are one machine."""
        parsed = urlparse(address)
        name = "loopback" if parsed.hostname in _LOOPBACK else (parsed.hostname or "")
        return parsed.scheme, name, parsed.port

    @classmethod
    def pool_spelling(cls, host: str) -> str | None:
        """The cluster's spelling of `host` if it is a node of this cluster, else None (same machine counts)."""
        wanted = cls.pool_key(host)
        return next((h for h in matricxon_pool.get_effective_hosts() if cls.pool_key(h) == wanted), None)

    async def _outside_cluster(self, config, host: str) -> bool:
        """A listed host the running cluster does not include. Unknown without consensus: counts as member."""
        if host in config.cluster_hosts or await self._seed_consensus(config) is None:
            return False
        return await self._consensus_status(host) is None

    async def check(self, address: str) -> NodeCheck:
        host = self.normalize(address)
        listed = self.pool_spelling(host)
        config = await settings_service.get_matricxon_server_config(self._db)
        if listed and not await self._outside_cluster(config, listed):
            return NodeCheck(
                address=host,
                reachable=True,
                in_cluster=True,
                reasons=["it is already a node of this cluster: remove it first, then add it again"],
            )
        async with matricxon_http.client(timeout=_TIMEOUT, transport=self._transport) as client:
            try:
                resp = await client.get(host + "/api/cluster/fresh")
            except httpx.HTTPError as exc:
                return NodeCheck(address=host, reachable=False, error=str(exc) or type(exc).__name__)
        if resp.status_code == 404:
            return NodeCheck(
                address=host, reachable=False, error="this Matricxon version has no cluster mode — update it"
            )
        if resp.is_error:
            return NodeCheck(address=host, reachable=False, error=f"HTTP {resp.status_code}")
        body = resp.json()
        return NodeCheck(address=host, reachable=True, fresh=body["fresh"], reasons=body["reasons"])

    async def reset(self, address: str, dropped: bool = False) -> None:
        """Resets a removed node to fresh (it restarts) so it can be re-added; only its own setup is cleared."""
        host = self.normalize(address)
        if self.pool_spelling(host):
            raise NodeNotAllowedError(f"{host} is still a node of this cluster: remove it first.")
        force = dropped  # `dropped`: already taken out of the voters
        if not dropped:
            config = await settings_service.get_matricxon_server_config(self._db)
            theirs = await self._consensus_status(host)
            force = bool(theirs and theirs.get("is_voter") and await self._seed_consensus(config) is not None)
            if force:  # still a voter but unlisted: remove it first
                await self.drop_unknown_voter(theirs["node_id"])
        async with matricxon_http.client(timeout=_TIMEOUT, transport=self._transport) as client:
            try:
                resp = await client.post(host + "/api/cluster/reset", params={"force": "true"} if force else None)
            except httpx.HTTPError as exc:
                raise NodeNotAllowedError(f"{host} is not reachable: {exc}") from exc
        if resp.status_code == 404:
            raise NodeNotAllowedError(f"{host} has no reset: update its Matricxon.")
        if resp.status_code == 409:
            raise NodeResetRefusedError(f"{host} refused: {resp.json().get('detail') or resp.json().get('error')}")
        if resp.is_error:
            raise NodeNotAllowedError(f"{host} could not be reset: HTTP {resp.status_code}")
        logger.info("cluster: node %s reset to a fresh install (restarting)", host)

    async def add(self, address: str, advertise_url: str | None = None, node_kind: str = "compute") -> str:
        """Provisions a fresh node with the saved folders and adds it to the pool; returns its address.
        `advertise_url`: how other nodes reach it, if not as pAIring does.
        """
        config = await settings_service.get_matricxon_server_config(self._db)
        host = self.normalize(address)
        advertise = self.normalize(advertise_url) if advertise_url and advertise_url.strip() else host
        if config.cluster_role == "off":
            raise NodeNotAllowedError("Turn cluster mode on first.")
        await self._adopt_paths(config)
        if not config.models_path or not config.prompt_cache_path:
            raise NodeNotAllowedError("Set the shared models and saved chat caches folders first.")
        listed = self.pool_spelling(host)
        if listed and not await self._outside_cluster(config, listed):
            raise NodeNotAllowedError(f"{host} is already a node of this cluster.")
        found = await self.check(host)
        if not found.reachable:
            raise NodeNotAllowedError(f"{host} is not reachable: {found.error}")
        if not found.fresh:
            raise NodeNotFreshError(f"{host} is not a fresh install: {'; '.join(found.reasons)}")
        voters = await self._voter_addresses(config)
        if voters:
            reach = ClusterReach(self._transport)
            await reach.cluster_reaches(await self._seed_consensus(config), host, advertise)
            await reach.joiner_reaches(host, voters)
        await self._provision(
            host, config.models_path, config.prompt_cache_path, advertise, voters, node_kind, config.admin_secret
        )
        listed = listed or host  # a listed plain host keeps its spelling
        config.remote_hosts = [h for h in config.remote_hosts if h != listed]
        config.cluster_hosts = [*config.cluster_hosts, listed]
        if node_kind == "voter":
            config.voter_hosts = [*config.voter_hosts, listed]
        await self._save(config)
        logger.info("cluster: node %s added (checked fresh, set up as a joiner, restarting)", host)
        return host

    async def remove(self, address: str) -> None:
        """Removes the node from the cluster and pool, then resets it (best effort; else use Reset to fresh).
        Added nodes can always go; others once consensus runs, if another node and voter remain.
        Removing the managed node switches pAIring to remote mode.
        """
        host = self.normalize(address)
        config = await settings_service.get_matricxon_server_config(self._db)
        hosts = matricxon_pool.get_effective_hosts()
        host = self.pool_spelling(host) or host
        if host not in hosts:
            raise NodeNotAllowedError(f"{host} is not a node of this cluster.")
        await self._adopt_paths(config)  # keep the cluster's folders if this was the node holding them
        entry = await self._seed_consensus(config)
        if entry is not None:
            await self._check_removable(entry, host, hosts)
        elif host not in config.cluster_hosts:
            raise NodeNotAllowedError("Removing this node needs the cluster to be running (consensus).")
        await self._drop_voter(entry, host)
        logger.info("cluster: node %s removed from the cluster", host)
        managed = host not in config.cluster_hosts and host not in config.remote_hosts
        ClusterHosts.without(config, host)
        if managed:
            logger.info(
                "cluster: pAIring no longer manages a local Matricxon (remote mode, hosts %s)", config.remote_hosts
            )
        await self._save(config)
        try:  # best effort: a bare leftover voter is worse than a manual reset
            await self.reset(host, dropped=True)
        except (NodeNotAllowedError, NodeResetRefusedError) as exc:
            logger.warning("cluster: removed node %s was not reset: %s", host, exc)

    async def _adopt_paths(self, config) -> None:
        """Fills missing shared folders from the running cluster (they live in its consensus config)."""
        if config.models_path and config.prompt_cache_path:
            return
        entry = await self._seed_consensus(config)
        paths = (entry[1].get("config") or {}).get("cluster/paths") or {} if entry else {}
        config.models_path = config.models_path or paths.get("models_dir") or None
        config.prompt_cache_path = config.prompt_cache_path or paths.get("prompt_cache_dir") or None

    async def _voter_addresses(self, config) -> dict[str, str] | None:
        seed = await self._seed_consensus(config)
        return dict(seed[1].get("members") or {}) if seed else None

    async def _seed_consensus(self, config) -> tuple[str, dict] | None:
        """A voter to ask (the leader if any): (address, status)."""
        return await ClusterSettings(self._db, self._transport).entry_voter()

    async def _consensus_status(self, host: str) -> dict | None:
        async with matricxon_http.client(timeout=_TIMEOUT, transport=self._transport) as client:
            try:
                resp = await client.get(host + "/api/consensus/status")
                body = resp.json() if resp.status_code == 200 else {}
            except (httpx.HTTPError, ValueError):
                return None
        return body if body.get("enabled") else None

    async def _post(self, url: str, body: dict) -> None:
        await self._request("POST", url, body)

    async def _request(self, method: str, url: str, body: dict | None = None) -> None:
        async with matricxon_http.client(timeout=httpx.Timeout(30.0, connect=5.0), transport=self._transport) as client:
            try:
                await client.request(method, url, json=body)
            except httpx.HTTPError:
                pass  # best effort: retried at the next Activate

    async def _provision(
        self,
        host: str,
        models: str,
        cache: str,
        advertise: str,
        voters: dict[str, str] | None,
        node_kind: str = "compute",
        admin_secret: str | None = None,
    ) -> None:
        body: dict = {"models_dir": models, "prompt_cache_dir": cache, "restart": True}
        if node_kind != "compute":
            body["node_kind"] = node_kind
        if admin_secret:
            body["admin_secret"] = admin_secret
        if voters:  # joiner learns its address and whom to catch up from
            body |= {"advertise_url": advertise, "consensus_seeds": voters}
        async with matricxon_http.client(timeout=_TIMEOUT, transport=self._transport) as client:
            try:
                resp = await client.post(host + "/api/cluster/provision", json=body)
            except httpx.HTTPError as exc:
                raise NodeNotAllowedError(f"Could not configure {host}: {exc}") from exc
        if resp.status_code == 409:
            raise NodeNotFreshError(f"{host}: {resp.json().get('detail', 'not a fresh install')}")
        if resp.is_error:
            detail = resp.json().get("detail", f"HTTP {resp.status_code}") if resp.content else resp.status_code
            raise NodeNotAllowedError(f"{host} refused the configuration: {detail}. Is the shared storage mounted?")

    async def _save(self, config) -> None:
        await settings_service.set_matricxon_server_config(self._db, config)
        matricxon_pool.refresh_from_config(config)
        await server_pool_broadcast.broadcast_refresh("/api/settings/matricxon/internal-refresh")
