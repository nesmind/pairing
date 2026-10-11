"""One node's address book, paged, for debugging. Read from its consensus status; cluster nodes only."""

import httpx

from app.schemas.cluster import PeerAddress, PeerList
from app.services.cluster_nodes import ClusterNodes, NodeNotAllowedError
from app.services.cluster_settings import ClusterSettings

MAX_PAGE = 500


class ClusterPeers:
    def __init__(self, db, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._db = db
        self._transport = transport

    async def of(self, address: str, query: str = "", offset: int = 0, limit: int = 200) -> PeerList:
        host = ClusterNodes.pool_spelling(ClusterNodes.normalize(address))
        if host is None:
            raise NodeNotAllowedError(f"{address} is not a node of this cluster.")
        status = await ClusterSettings(self._db, self._transport).status(host)
        limit = max(1, min(limit, MAX_PAGE))
        if status is None:
            return PeerList(node=host, total=0, offset=0, limit=limit, peers=[])
        rows = self._rows(status)
        wanted = query.strip().lower()
        if wanted:
            rows = [r for r in rows if wanted in r.id.lower() or wanted in r.address.lower()]
        offset = max(0, offset)
        return PeerList(
            node=host,
            advertise_url=status.get("advertise_url"),
            total=len(rows),
            offset=offset,
            limit=limit,
            peers=rows[offset : offset + limit],
        )

    @staticmethod
    def _rows(status: dict) -> list[PeerAddress]:
        """One row per address: current nodes first, then remembered ones; older identities are counted in the row."""
        voters, seeds, me = set(status.get("voters", [])), status.get("seeds", {}), status.get("node_id")
        addresses = status.get("addresses", {})
        nodes = dict(status.get("nodes") or {}) or {i: addresses[i] for i in voters if i in addresses}
        rows = [
            PeerAddress(
                id=i,
                address=a,
                voter=i in voters,
                this_node=i == me,
                seed=i in seeds,
                kind="voter" if i in voters else "learner",
            )
            for i, a in nodes.items()
        ]
        shown = {a.rstrip("/"): row for (i, a), row in zip(nodes.items(), rows, strict=True)}
        for i, a in addresses.items():
            if i in nodes:
                continue
            row = shown.get(a.rstrip("/"))
            if row is not None:
                row.old_ids += 1
                continue
            row = PeerAddress(id=i, address=a, voter=False, seed=i in seeds, kind="old")
            shown[a.rstrip("/")] = row
            rows.append(row)
        order = {"voter": 0, "learner": 1, "old": 2}
        return sorted(rows, key=lambda r: (not r.this_node, order[r.kind], r.id))
