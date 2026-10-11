"""Mirrors the cluster's own member list into pAIring's node list (primary instance only).

Each node reports `nodes` (voters plus learners) and `removed` in its consensus status. This adds
unlisted nodes and drops removed ones. Unanswering listed nodes are kept. Nodes match by id, so
pAIring's spelling of an address is kept. pAIring never changes who votes.
"""

import asyncio
import logging

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AsyncSessionLocal
from app.services import matricxon_pool, server_pool_broadcast, settings_service
from app.services.cluster_hosts import ClusterHosts
from app.services.cluster_nodes import ClusterNodes
from app.services.cluster_settings import ClusterSettings

logger = logging.getLogger("llama_chat")
INTERVAL = 15.0


class ClusterMembership:
    def __init__(self) -> None:
        # listed host -> address the cluster knows it by (so a restarting node is not listed twice)
        self._advertised: dict[str, str] = {}

    async def sync(self, db: AsyncSession, transport: httpx.AsyncBaseTransport | None = None) -> bool:
        """One look; True when the list changed."""
        config = await settings_service.get_matricxon_server_config(db)
        if config.cluster_role == "off":
            return False
        listed = ClusterHosts.nodes(config)
        settings = ClusterSettings(db, transport, listed)
        statuses = {
            h: s for h, s in zip(listed, await asyncio.gather(*(settings.status(h) for h in listed)), strict=True) if s
        }
        if not statuses:
            return False
        cluster = max(statuses.values(), key=lambda s: (s.get("epoch", 0), s.get("applied_slot", 0)))
        nodes, removed = cluster.get("nodes") or {}, set(cluster.get("removed") or [])
        if not nodes:
            return False  # older Matricxon without a node list
        by_id = {s["node_id"]: h for h, s in statuses.items()}
        # What each listed node reported is remembered (also across restarts): a node that is restarting is still
        # recognised, instead of being added again under the address the cluster knows it by.
        remembered = {h: i for h, i in config.known_nodes.items() if h in listed}
        for h, s in statuses.items():
            remembered[h] = {"node_id": s["node_id"], "advertise_url": s.get("advertise_url") or ""}
        known_ids = {i["node_id"] for i in remembered.values()}
        self._advertised.update(
            {h: s["advertise_url"].rstrip("/") for h, s in statuses.items() if s.get("advertise_url")}
        )
        gone = [h for h, s in statuses.items() if s["node_id"] in removed]
        gone += self._duplicates(listed, statuses, config.first_node)
        add, twins = [], []
        for node_id, address in nodes.items():
            address = address.rstrip("/")
            if node_id in by_id and address in listed and address not in statuses and address != by_id[node_id]:
                twins.append(address)  # cluster's name for a node listed under another, unreachable one
            elif (
                node_id not in by_id
                and node_id not in known_ids
                and node_id not in removed
                and not self._listed(listed, address)
            ):
                add.append(address)
        gone += twins
        if not gone and not add:
            if remembered != config.known_nodes:
                config.known_nodes = remembered
                await settings_service.set_matricxon_server_config(db, config)
            return False
        config.known_nodes = remembered
        for host in gone:
            ClusterHosts.without(config, host)
        config.cluster_hosts = [*config.cluster_hosts, *add]
        await settings_service.set_matricxon_server_config(db, config)
        matricxon_pool.refresh_from_config(config)
        await server_pool_broadcast.broadcast_refresh("/api/settings/matricxon/internal-refresh")
        logger.info("cluster: node list follows the cluster (added %s, dropped %s)", add or "none", gone or "none")
        return True

    @staticmethod
    def _duplicates(listed: list[str], statuses: dict[str, dict], first_node: str | None) -> list[str]:
        """Hosts that answer with the id of a node already listed (the same node under another address, e.g. its
        LAN address next to 127.0.0.1): the first listed one stays, the chosen first node always.
        """
        by_id: dict[str, list[str]] = {}
        for host in listed:
            if host in statuses:
                by_id.setdefault(statuses[host]["node_id"], []).append(host)
        extra: list[str] = []
        for hosts in by_id.values():
            keep = first_node if first_node in hosts else hosts[0]
            extra += [h for h in hosts if h != keep]
        return extra

    def _listed(self, listed: list[str], address: str) -> bool:
        """Already listed: same spelling, same machine and port, or a known advertised address."""
        wanted = ClusterNodes.pool_key(address)
        known = {self._advertised[h] for h in listed if h in self._advertised}
        return address in known or any(ClusterNodes.pool_key(h) == wanted for h in listed)


_membership = ClusterMembership()
_tasks: set[asyncio.Task] = set()


async def _run_forever() -> None:
    while True:
        try:
            async with AsyncSessionLocal() as db:
                await _membership.sync(db)
        except Exception:  # noqa: BLE001 - one bad look must not end the loop
            logger.exception("cluster: following the cluster's node list failed")
        await asyncio.sleep(INTERVAL)


def start_membership_sync() -> None:
    """Called once from startup_service.run_startup_tasks (primary only)."""
    task = asyncio.create_task(_run_forever())
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
