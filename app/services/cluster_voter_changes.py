"""Changing who votes: demote, promote a joiner, drop a voter. Mixed into ClusterNodes (uses its
helpers `pool_spelling`, `_seed_consensus`, `_consensus_status`, `_post`). Goes through the leader; needs a quorum.
"""

import logging

import httpx

from app.services import matricxon_http, matricxon_pool, settings_service
from app.services.cluster_errors import NodeNotAllowedError

logger = logging.getLogger("llama_chat")


class VoterChanges:
    async def demote(self, address: str) -> None:
        """A voter becomes a learner (serves chats, replays the log). The last voter cannot be demoted."""
        host = self.pool_spelling(self.normalize(address))
        if host is None:
            raise NodeNotAllowedError(f"{address} is not a node of this cluster.")
        config = await settings_service.get_matricxon_server_config(self._db)
        entry = await self._seed_consensus(config)
        if entry is None:
            raise NodeNotAllowedError("This needs the cluster to be running (consensus).")
        theirs = await self._consensus_status(host)
        voters = entry[1]["voters"]
        if not theirs or theirs["node_id"] not in voters:
            raise NodeNotAllowedError(f"{host} is not a voter.")
        if len(voters) <= 1:
            raise NodeNotAllowedError(f"{host} is the cluster's only voter: it cannot stop voting.")
        target = entry[1].get("target_voters", 0)  # the cluster keeps this many itself (0: none)
        if target and len(voters) <= target:
            raise NodeNotAllowedError(f"The cluster keeps {target} voters on its own: it has {len(voters)}.")
        async with matricxon_http.client(timeout=httpx.Timeout(30.0, connect=5.0), transport=self._transport) as client:
            try:
                resp = await client.delete(f"{entry[0]}/api/consensus/voters/{theirs['node_id']}")
            except httpx.HTTPError as exc:
                raise NodeNotAllowedError(f"Could not reach {entry[0]}: {exc}") from exc
        if resp.is_error or not resp.json().get("ok"):
            raise NodeNotAllowedError("The cluster did not make the change in time (is there a quorum?).")
        logger.info("cluster: node %s is no longer a voter (kept in the cluster as a learner)", host)

    async def _check_removable(self, entry: tuple[str, dict], host: str, hosts: list[str]) -> None:
        """Never allowed, however the node joined: the last node or last voter leaving."""
        if len(hosts) < 2:
            raise NodeNotAllowedError(f"{host} is the only node: add another one first.")
        theirs = await self._consensus_status(host)
        voters = entry[1]["voters"]
        if theirs and theirs["node_id"] in voters and len(voters) == 1:
            raise NodeNotAllowedError(f"{host} is the cluster's only voter: make another node a voter first.")

    # consensus: a joiner becomes a voter

    CATCH_UP_SLACK = 50  # slots a learner may lag and be made a voter

    async def ensure_voters(self) -> str | None:
        """Makes every consensus node a voter once caught up (called on Activate). A node far behind is
        left for the next Activate and named in the returned message.
        """
        config = await settings_service.get_matricxon_server_config(self._db)
        seed = await self._seed_consensus(config)
        if seed is None:
            return None
        seed_host, status = seed
        waiting: list[str] = []
        for joiner in matricxon_pool.get_effective_hosts():
            theirs = await self._consensus_status(joiner)
            if not theirs:
                continue
            is_voter = theirs["node_id"] in status["voters"]
            if is_voter and not self.address_moved(theirs, status):
                continue
            behind = (
                status.get("epoch", 0) - theirs.get("epoch", 0),
                status.get("applied_slot", 0) - theirs.get("applied_slot", 0),
            )
            if not is_voter and (behind[0] > 0 or behind[1] > self.CATCH_UP_SLACK):
                waiting.append(joiner)
                logger.info("cluster: node %s is still catching up, not made a voter yet", joiner)
                continue
            # The address the OTHER nodes must use: what the node advertises, not pAIring's spelling.
            address = theirs.get("advertise_url") or joiner
            await self._post(seed_host + "/api/consensus/voters", {"id": theirs["node_id"], "address": address})
        if waiting:
            return f"Still catching up with the cluster; make it a voter again in a minute: {', '.join(waiting)}"
        return None

    @staticmethod
    def address_moved(theirs: dict, cluster: dict) -> bool:
        """A voter advertising a different address than recorded (the others use the old one)."""
        advertised = (theirs.get("advertise_url") or "").rstrip("/")
        recorded = (cluster.get("members", {}).get(theirs["node_id"]) or "").rstrip("/")
        return bool(advertised and recorded and advertised != recorded)

    async def _drop_voter(self, entry: tuple[str, dict] | None, host: str) -> None:
        """Removes the node from the voters and records it as removed so it is not promoted again
        (covers every id at this address). If the cluster refuses, the node stays in pAIring too.
        """
        if entry is None:
            return
        theirs = await self._consensus_status(host)
        wanted = self.pool_key(host)
        ids = {theirs["node_id"]} if theirs else set()
        ids |= {i for i, address in entry[1].get("nodes", {}).items() if self.pool_key(address) == wanted}
        for node_id in sorted(ids):
            await self._delete_voter(entry[0], node_id, f"the cluster did not take {host} out (a quorum?)")

    async def _delete_voter(self, via: str, node_id: str, failure: str) -> None:
        async with matricxon_http.client(timeout=httpx.Timeout(30.0, connect=5.0), transport=self._transport) as client:
            try:
                resp = await client.delete(f"{via}/api/consensus/voters/{node_id}", params={"leave": "true"})
            except httpx.HTTPError as exc:
                raise NodeNotAllowedError(f"Could not reach {via}: {exc}") from exc
        if resp.is_error or not resp.json().get("ok"):
            raise NodeNotAllowedError(failure)

    async def drop_unknown_voter(self, node_id: str) -> None:
        """Removes a voter no listed node matches. Needs a quorum and keeps one voter."""
        config = await settings_service.get_matricxon_server_config(self._db)
        entry = await self._seed_consensus(config)
        if entry is None:
            raise NodeNotAllowedError("This needs the cluster to be running (consensus).")
        voters = entry[1]["voters"]
        if node_id not in voters:
            raise NodeNotAllowedError("That is not a voter of the cluster.")
        if len(voters) <= 1:
            raise NodeNotAllowedError("It is the cluster's only voter: it cannot be removed.")
        for host in matricxon_pool.get_effective_hosts():
            theirs = await self._consensus_status(host)
            if theirs and theirs["node_id"] == node_id:
                raise NodeNotAllowedError(f"{host} is that voter: remove it as a node instead.")
        await self._delete_voter(entry[0], node_id, "The cluster did not take it out (is there a quorum?).")
        logger.info("cluster: voter %s (not in the host list) was taken out of the voters", node_id[:8])
