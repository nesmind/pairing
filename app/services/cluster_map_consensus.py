"""Consensus part of the map: voter links (star from the leader) and an amber box for unmatched voters."""

from urllib.parse import urlparse

from app.schemas.cluster import ClusterHost, MapLink, MapNode


def short_address(address: str) -> str:
    return urlparse(address).netloc or address


class ConsensusLinks:
    def of(self, hosts: list[ClusterHost], statuses: dict[str, dict]) -> tuple[list[MapNode], list[MapLink]]:
        """Voter links and amber boxes for voters no host matches."""
        by_consensus_id = {s["node_id"]: host for host, s in statuses.items()}
        voters: dict[str, str | None] = {}
        newest = max((s.get("epoch", 0) for s in statuses.values()), default=0)
        for status in statuses.values():
            if status.get("epoch", 0) < newest:
                continue  # a replaying node lists long-gone voters
            for voter in status["voters"]:
                voters.setdefault(voter, status.get("members", {}).get(voter))
        ghosts, ids = [], {}
        for voter, address in voters.items():
            if voter in by_consensus_id:
                ids[voter] = by_consensus_id[voter]
            elif address in {h.host for h in hosts}:
                ids[voter] = address
            else:
                ids[voter] = f"voter:{voter}"
                ghosts.append(
                    MapNode(
                        id=f"voter:{voter}",
                        kind="voter",
                        name=f"Voter {voter}",
                        subtitle=short_address(address) if address else "not in the node list",
                        status="warn",  # not known down: pAIring has no host to read it
                        badges=["voter"],
                        removable=len(voters) > 1,
                        address=address,
                        details={
                            "Voter id": voter,
                            "Address": address or "unknown",
                            "Note": "a voter of the cluster that is not one of pAIring's nodes (the node list "
                            "is managed by the cluster). Take it out of the voters below, or use Reset to "
                            "fresh on the Add nodes page, which does that first.",
                        },
                    )
                )
        leader_ids = [s["node_id"] for s in statuses.values() if s["role"] == "leader"]
        links = []
        for source, target in self._pairs(list(ids), leader_ids):
            links.append(
                MapLink(
                    source=ids[source],
                    target=ids[target],
                    kind="consensus",
                    active=not str(ids[target]).startswith("voter:"),
                    leader=source in leader_ids,
                )
            )
        return ghosts, links

    @staticmethod
    def _pairs(voters: list[str], leaders: list[str]) -> list[tuple[str, str]]:
        """Star from the leader if any, else every pair once."""
        if leaders:
            return [(leaders[0], v) for v in voters if v != leaders[0]]
        return [(a, b) for i, a in enumerate(voters) for b in voters[i + 1 :]]
