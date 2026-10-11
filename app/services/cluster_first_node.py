"""The node that starts the cluster: chosen on the Cluster page and saved in config."""

from app.schemas import MatricxonServerConfig
from app.services.cluster_hosts import ClusterHosts
from app.services.cluster_nodes import ClusterNodes


class FirstNode:
    @staticmethod
    def choose(config: MatricxonServerConfig, value: str | None) -> None:
        """Saves the choice: None keeps, blank clears, else any address (a new one is added to pAIring's hosts)."""
        if value is None:
            return
        value = value.strip()
        if not value:
            config.first_node = None
            return
        address = ClusterNodes.normalize(value)
        host = ClusterNodes.pool_spelling(address)
        if host is None:  # not known to pAIring yet: it becomes one of its (remote) hosts
            nodes = ClusterHosts.nodes(config)
            config.remote_hosts = [h for h in nodes if h not in config.cluster_hosts] + [address]
            config.mode = "remote"
            host = address
        config.first_node = host

    @staticmethod
    def required(config: MatricxonServerConfig) -> str:
        if not config.first_node or config.first_node not in ClusterHosts.nodes(config):
            raise ValueError("Choose the first node and save it with the shared folders first.")
        return config.first_node
