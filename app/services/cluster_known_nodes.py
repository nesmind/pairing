"""Remembers, across restarts, which node id and advertised address each listed host reported.

A stopped node cannot say who it is, and the cluster knows it by its advertised address (e.g. its LAN IP), not the
host pAIring lists it under; without this memory the Nodes map would show it twice.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.cluster import ClusterHost
from app.services import settings_service


class KnownNodes:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def update(self, hosts: list[ClusterHost]) -> dict[str, dict[str, str]]:
        """Returns {host: {"node_id", "advertise_url"}} for every listed host seen so far; saves only on a change."""
        config = await settings_service.get_matricxon_server_config(self._db)
        known = {h: v for h, v in config.known_nodes.items() if h in {x.host for x in hosts}}
        for host in hosts:
            seen = host.consensus
            if host.reachable and seen and seen.get("node_id"):
                known[host.host] = {"node_id": seen["node_id"], "advertise_url": seen.get("advertise_url") or ""}
        if known != config.known_nodes:
            config.known_nodes = known
            await settings_service.set_matricxon_server_config(self._db, config)
        return known
