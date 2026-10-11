"""The nodes pAIring talks to, as one list.

Stored in `remote_hosts` (remote mode) and `cluster_hosts` (Cluster page); in local mode
`LOCAL_MATRICXON_HOST` comes first. Only this module knows that.
"""

from app.schemas import MatricxonServerConfig
from app.services.matricxon_pool import LOCAL_MATRICXON_HOST


class ClusterHosts:
    @staticmethod
    def nodes(config: MatricxonServerConfig) -> list[str]:
        base = config.remote_hosts if config.mode == "remote" else [LOCAL_MATRICXON_HOST]
        return base + [h for h in config.cluster_hosts if h not in base]

    @classmethod
    def without(cls, config: MatricxonServerConfig, host: str) -> None:
        """Removes `host` wherever stored. Removing the managed node switches to remote mode."""
        left = [h for h in cls.nodes(config) if h != host]
        stored = host in config.cluster_hosts or host in config.remote_hosts
        config.cluster_hosts = [h for h in config.cluster_hosts if h != host]
        config.voter_hosts = [h for h in config.voter_hosts if h != host]
        config.first_node = None if config.first_node == host else config.first_node
        config.remote_hosts = [h for h in config.remote_hosts if h != host]
        if not stored:
            config.mode, config.remote_hosts, config.cluster_hosts = "remote", left, []
        elif config.mode == "remote" and not config.remote_hosts and config.cluster_hosts:
            # An empty remote list would fall back to local mode.
            config.remote_hosts, config.cluster_hosts = config.cluster_hosts, []

    @classmethod
    def keep_through_a_save(cls, previous: MatricxonServerConfig, saved: MatricxonServerConfig) -> None:
        """Keeps the cluster settings out of the Matricxon form: with cluster mode on, nodes and
        shared folders belong to the Cluster page. Switching local/remote keeps every node.
        """
        # Owned by the Cluster page; a stale form save must not wipe them (losing the advertise
        # address once cost a restarted seed its quorum).
        saved.cluster_hosts = previous.cluster_hosts
        saved.voter_hosts = previous.voter_hosts
        saved.admin_secret = previous.admin_secret
        saved.advertise_url = previous.advertise_url
        saved.first_node = previous.first_node
        saved.cluster_role = previous.cluster_role
        if previous.cluster_role == "off":
            return  # no cluster: the form's values are real
        saved.models_path = previous.models_path
        saved.prompt_cache_path = previous.prompt_cache_path
        nodes = cls.nodes(previous)
        if saved.mode == "remote":
            saved.remote_hosts = [h for h in nodes if h not in previous.cluster_hosts]
        else:
            saved.remote_hosts = []
            saved.cluster_hosts = [h for h in nodes if h != LOCAL_MATRICXON_HOST]
