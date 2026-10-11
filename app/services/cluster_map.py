"""The Nodes map: every server of the cluster, its state and links, built live from what each reports."""

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.cluster import ClusterHost, ClusterMap, ClusterState, MapLink, MapNode
from app.services import matricxon_pool
from app.services.cluster_known_nodes import KnownNodes
from app.services.cluster_map_consensus import ConsensusLinks, short_address
from app.services.cluster_map_storage import StorageNode
from app.services.cluster_nodes import ClusterNodes
from app.services.cluster_replay import LogReplay
from app.services.cluster_service import ClusterService
from app.services.cluster_settings import ClusterSettings

_TIMEOUT = httpx.Timeout(3.0, connect=1.5)
_FANOUT = 32
_DETAIL_SERVERS = 6  # up to this many servers: detailed diagram (cluster_map.js)
_LEASE_BADGES = {"model_pull": "pulls", "cache_clean": "cleans"}


_short = short_address
_storage = StorageNode()
_consensus_links = ConsensusLinks()


class ClusterMapBuilder:
    def __init__(
        self, db: AsyncSession, service: ClusterService, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self._db = db
        self._service = service
        self._transport = transport

    async def build(self, compact: bool = False) -> ClusterMap:
        """The whole map. `compact` keeps only what is drawn and voter links; details load on click (`node`)."""
        state = await self._service.state()
        if not state.supported:
            return ClusterMap(enabled=state.enabled, supported=False, nodes=[], links=[])
        known = await KnownNodes(self._db).update(state.hosts)
        return self._map(state, self._statuses(state.hosts), compact, known)

    async def node(self, node_id: str) -> MapNode | None:
        """One node's full record (one readiness + one status call)."""
        if node_id in matricxon_pool.get_effective_hosts():
            state = await self._service.state(only=[node_id])
            host = state.hosts[0] if state.hosts else None
            if host is None:
                return None
            # Removal depends on the whole cluster's size.
            size = len(matricxon_pool.get_effective_hosts())
            log_total = await self._leader_total(host.consensus)
            return self._matricxon_node(host, state, host.consensus, cluster_size=size, log_total=log_total)
        full = await self.build()
        return next((n for n in full.nodes if n.id == node_id), None)

    async def _leader_total(self, consensus: dict | None) -> int:
        """The log length as the leader reports it (one call, only for a follower)."""
        if not consensus or consensus.get("is_voter") or not consensus.get("enabled"):
            return 0
        address = ClusterNodes.pool_spelling(consensus.get("members", {}).get(consensus.get("leader"), ""))
        status = await ClusterSettings(self._db, self._transport).status(address) if address else None
        return LogReplay.total([status] if status else [])

    @staticmethod
    def _statuses(hosts: list[ClusterHost]) -> dict[str, dict]:
        return {h.host: h.consensus for h in hosts if h.consensus}

    def _map(self, state, consensus: dict, compact: bool, known: dict | None = None) -> ClusterMap:
        nodes = [self._app_node(state)]
        links: list[MapLink] = []
        log_total = LogReplay.total(list(consensus.values()))
        members = {v for status in consensus.values() for v in (*status.get("voters", []), *status.get("nodes", {}))}
        inside = [
            h
            for h in state.hosts
            if self._in_cluster(h, state, consensus) or self._was_member(h, consensus, known or {}, members)
        ]
        outside = [h.host for h in state.hosts if h not in inside]
        matricxon = [self._matricxon_node(h, state, consensus.get(h.host), log_total=log_total) for h in inside]
        nodes += matricxon
        nodes.append(_storage.node(state, inside))
        links += [MapLink(source="app", target=n.id, kind="api", active=n.status != "down") for n in matricxon]
        links += [MapLink(source=n.id, target="storage", kind="storage", active=n.status == "up") for n in matricxon]
        ghosts, consensus_links = _consensus_links.of(inside, consensus, known)
        nodes += ghosts
        links += consensus_links
        if compact:
            nodes = [n.model_copy(update={"details": {}, "checks": []}) for n in nodes]
            if len(matricxon) > _DETAIL_SERVERS:  # small map: every link; big: voters only
                links = [link for link in links if link.kind == "consensus"]
        return ClusterMap(
            quorum=state.quorum, enabled=state.enabled, supported=True, nodes=nodes, links=links, outside=outside
        )

    @staticmethod
    def _was_member(host: ClusterHost, consensus: dict, known: dict, members: set[str]) -> bool:
        """A stopped node that reported a node id the cluster still counts: drawn (down), not hidden."""
        return host.host not in consensus and known.get(host.host, {}).get("node_id") in members

    @staticmethod
    def _in_cluster(host: ClusterHost, state: ClusterState, consensus: dict) -> bool:
        """Nodes that run the cluster, were added or are being set up. Hosts merely listed are not drawn."""
        if not state.enabled:
            return False  # no cluster: nothing to draw
        if state.quorum == "off" and not state.paths.first_node:
            return True  # setup not begun: every host
        mine = consensus.get(host.host)
        return (
            bool(mine and mine.get("enabled")) or host.host in state.node_hosts or host.host == state.paths.first_node
        )

    # nodes

    def _app_node(self, state: ClusterState) -> MapNode:
        failing = [c for c in state.app_checks if c.status != "ok"]
        return MapNode(
            id="app",
            kind="app",
            name="pAIring",
            subtitle="control",
            status="warn" if failing else "up",
            details={"Role": "control plane (this app)", "Cluster mode": "on" if state.enabled else "off"},
            checks=state.app_checks,
        )

    @staticmethod
    def _address_details(consensus: dict) -> dict[str, str]:
        """The address that matters at a glance; the full book is at GET /api/cluster/peers."""
        return {"Advertises": consensus.get("advertise_url") or "not set"}

    @staticmethod
    def _address_mismatch(consensus: dict) -> str | None:
        """Set when a voter advertises a different address than recorded."""
        if not consensus.get("is_voter"):
            return None
        if not ClusterNodes.address_moved(consensus, {"members": consensus.get("members", {})}):
            return None
        recorded = consensus["members"][consensus["node_id"]]
        return f"advertises {consensus['advertise_url']}, the cluster uses {recorded} (press 'Update its address')"

    @staticmethod
    def _is_learner(consensus: dict | None) -> bool:
        """Runs consensus but does not vote yet."""
        return bool(consensus and consensus.get("enabled") and not consensus.get("is_voter"))

    @staticmethod
    def _can_demote(consensus: dict | None) -> bool:
        """A voter may step down while above the cluster's own target."""
        if not consensus or not consensus.get("is_voter"):
            return False
        return len(consensus["voters"]) > max(consensus.get("target_voters", 0), 1)

    @staticmethod
    def _can_remove(host: ClusterHost, size: int, consensus_on: bool, added: bool) -> bool:
        """Before consensus: added nodes can go, the first cannot. After: any node, except the last
        node or last voter (as ClusterNodes.remove).
        """
        if not consensus_on:
            return added
        if size < 2:
            return False
        mine = host.consensus
        return not (mine and mine.get("is_voter") and len(mine.get("voters", [])) == 1)

    @staticmethod
    def _action_notes(host: ClusterHost, consensus: dict | None, size: int, ok: dict[str, bool]) -> dict[str, str]:
        """Why each node-panel action is unavailable."""
        sole_voter = bool(consensus and consensus.get("is_voter") and len(consensus.get("voters", [])) == 1)
        notes: dict[str, str] = {}
        if not ok["activate"]:
            if not host.reachable:
                notes["activate"] = "the node does not answer"
            elif not host.ready:
                notes["activate"] = "its checklist has items to fix"
            else:
                notes["activate"] = "it is already active in the cluster"
        if not ok["demote"]:
            if sole_voter:
                notes["demote"] = "it is the cluster's only voter: make another node a voter first"
            elif consensus and consensus.get("is_voter"):
                notes["demote"] = f"the cluster keeps {consensus.get('target_voters')} voters on its own"
            else:
                notes["demote"] = "it is not a voter" if consensus else "the cluster does not run consensus yet"
        if not ok["remove"]:
            if not consensus:
                notes["remove"] = "the cluster does not run consensus yet"
            elif size < 2:
                notes["remove"] = "it is the only node"
            elif sole_voter:
                notes["remove"] = "it is the cluster's only voter: make another node a voter first"
            else:
                notes["remove"] = "not available right now"
        return notes

    def _matricxon_node(
        self,
        host: ClusterHost,
        state: ClusterState,
        consensus: dict | None,
        cluster_size: int | None = None,
        log_total: int = 0,
    ) -> MapNode:
        size = cluster_size or len(state.hosts)
        consensus_on = bool(consensus and consensus.get("enabled")) or any(h.consensus for h in state.hosts)
        added = host.host in state.node_hosts
        badges = ["joiner" if added else "seed"] if host.reachable and state.enabled else []
        details: dict[str, object] = {"Address": host.host}
        status = "down"
        if host.reachable:
            status = "up" if host.ready else "warn"
            details |= {"Cluster role": host.role, "Node id": host.node_id}
            badges += ["active"] if host.active else []
        else:
            details["Error"] = host.error or "unreachable"
        if consensus_on and host.reachable and not (consensus and consensus.get("enabled")):
            # Plain host (e.g. kept by a reset): the cluster runs without it.
            badges = ["not in cluster"]
            details["Cluster role"] = "not in the cluster: add it under Add nodes"
        if consensus and consensus.get("enabled"):
            badges = [b for b in badges if b not in ("seed", "joiner")]
            badges += ["voter"] if consensus["is_voter"] else ["learner"]
            badges += ["leader"] if consensus["role"] == "leader" else []
            if consensus.get("node_kind") == "voter":
                badges.append("voter only")
                details["Node type"] = "voter only: never loads models or gets chats"
            # Not the .env setup label: who started the cluster no longer matters.
            details["Cluster role"] = (
                "leader" if consensus["role"] == "leader" else "voter" if consensus["is_voter"] else "learner"
            )
            if consensus.get("is_voter") and not consensus.get("has_quorum", True):
                badges.append("no quorum")
                status = "warn" if status == "up" else status
            badges += [_LEASE_BADGES[n] for n in consensus.get("holds", []) if n in _LEASE_BADGES]
            replay = LogReplay.progress(consensus, log_total)
            if replay:  # amber until it has the whole log
                percent = LogReplay.percent(*replay)
                badges.append(f"replaying the log {percent}%")
                status = "warn" if status == "up" else status
                details["Replaying the log"] = f"{replay[0]} of {replay[1]} entries ({percent}%): OK when it is done"
            details |= self._address_details(consensus)
            mismatch = self._address_mismatch(consensus)
            if mismatch:
                details["Address mismatch"] = mismatch
            details |= {
                "Consensus": f"{consensus['role']} (epoch {consensus['epoch']}, applied {consensus['applied_slot']})",
                "Voters": ", ".join(consensus["voters"]),
            }
        ok = {
            "activate": host.reachable
            and ("not in cluster" in badges or (host.ready and (not host.active or self._is_learner(consensus)))),
            "remove": self._can_remove(host, size, consensus_on, added),
            "demote": self._can_demote(consensus),
        }
        return MapNode(
            id=host.host,
            kind="matricxon",
            name="Matricxon",
            subtitle=_short(host.host),
            status=status,
            badges=badges,
            address=host.host,
            activatable=ok["activate"],
            removable=ok["remove"],
            demotable=ok["demote"],
            address_mismatch=self._address_mismatch(consensus) if consensus else None,
            action_notes=self._action_notes(host, consensus, size, ok),
            details=details,
            checks=host.checks,
        )
