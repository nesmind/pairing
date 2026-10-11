"""Before adding a node both directions must work: cluster -> new node's advertised address, and
new node -> a voter at its recorded address.
"""

import httpx

from app.services import matricxon_http
from app.services.cluster_errors import NodeNotAllowedError

_TIMEOUT = httpx.Timeout(20.0, connect=5.0)


class ClusterReach:
    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._transport = transport

    async def cluster_reaches(self, entry: tuple[str, dict] | None, host: str, advertise: str) -> None:
        """Other nodes must reach the new one at its advertised address, or it never catches up.
        A voter says WHO answers there. Checked before anything changes.
        """
        if entry is None:
            return
        async with matricxon_http.client(timeout=_TIMEOUT, transport=self._transport) as client:
            try:
                theirs = await client.get(host + "/api/cluster/readiness")
                reach = await client.post(entry[0] + "/api/consensus/reach", json={"address": advertise})
            except httpx.HTTPError:
                return
        if reach.status_code == 404 or theirs.status_code != 200:
            return  # older Matricxon cannot say
        found = reach.json()
        if found.get("reachable") and found.get("node_id") == theirs.json().get("node_id"):
            return
        raise NodeNotAllowedError(
            f"The cluster cannot reach {host} at {advertise}"
            + (f" ({found.get('detail')})" if found.get("detail") else " (another node answers there)")
            + ". Fill in the address the OTHER nodes use to reach it (e.g. its real IP, not 127.0.0.1)."
        )

    async def joiner_reaches(self, host: str, voters: dict[str, str]) -> None:
        """The new node must reach a voter at its recorded address, or it can never fetch the log
        (e.g. voter listens on 127.0.0.1 but advertises its LAN address). Asked of the new node.
        """
        async with matricxon_http.client(timeout=_TIMEOUT, transport=self._transport) as client:
            try:
                ours = (await client.get(host + "/api/cluster/readiness")).json().get("node_id")
                replies = [
                    await client.post(host + "/api/consensus/reach", json={"address": a}) for a in voters.values()
                ]
            except (httpx.HTTPError, ValueError):
                return
        if any(r.status_code == 404 for r in replies):
            return  # older Matricxon cannot say
        found = [r.json() for r in replies]
        if any(f.get("reachable") and f.get("node_id") != ours for f in found):
            return
        where = ", ".join(voters.values())
        raise NodeNotAllowedError(
            f"{host} cannot reach the cluster's node(s) at {where}, so it could not fetch the log. "
            "The first node must advertise an address the others can reach (not 127.0.0.1; from a container e.g. "
            "http://host.containers.internal:8441): reset the cluster and set it under 'Address the other nodes use to "
            "reach it' when choosing the first node."
        )
