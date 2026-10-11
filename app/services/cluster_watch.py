"""Logs cluster changes seen between two readings: node down/up, ready/active, host list changes,
leader moves, quorum lost/back. Costs nothing extra.
"""

import logging

from app.schemas.cluster import ClusterHost

logger = logging.getLogger("llama_chat")


class ClusterWatch:
    def __init__(self) -> None:
        self._hosts: dict[str, ClusterHost] | None = None
        self._quorum: str | None = None
        self._leader: str | None = None

    def observe(self, hosts: list[ClusterHost], quorum: str) -> None:
        now = {h.host: h for h in hosts}
        before, self._hosts = self._hosts, now
        old_quorum, self._quorum = self._quorum, quorum
        leader = next((h.host for h in hosts if h.consensus and h.consensus.get("role") == "leader"), None)
        old_leader, self._leader = self._leader, leader
        if before is None:
            return
        for host in before.keys() - now.keys():
            logger.info("cluster: node %s left the host list", host)
        for host, node in now.items():
            self._node(host, before.get(host), node)
        if quorum != old_quorum and "off" not in (quorum, old_quorum):
            if quorum == "lost":
                logger.warning("cluster: NO QUORUM: cluster changes are locked, chats keep working")
            else:
                logger.info("cluster: quorum is back")
        if leader and leader != old_leader:
            logger.info("cluster: the leader is now %s", leader)

    @staticmethod
    def _node(host: str, old: ClusterHost | None, node: ClusterHost) -> None:
        if old is None:
            logger.info("cluster: node %s joined the host list", host)
        elif old.reachable and not node.reachable:
            logger.warning("cluster: node %s is not responding (%s)", host, node.error or "unreachable")
        elif not old.reachable and node.reachable:
            logger.info("cluster: node %s is responding again", host)
        if node.reachable and old is not None and old.reachable:
            if node.active and not old.active:
                logger.info("cluster: node %s is now active in the cluster", host)
            elif node.ready and not old.ready:
                logger.info("cluster: node %s is now ready", host)
            elif old.ready and not node.ready:
                logger.warning("cluster: node %s is no longer ready (a check failed)", host)
