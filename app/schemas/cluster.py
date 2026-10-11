from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ClusterCheck(BaseModel):
    """One row of Matricxon's cluster-readiness checklist."""

    id: str
    status: str
    what: str
    why: str
    setting: str | None = None
    size_bytes: int = 0
    hint: str = ""


class ClusterHost(BaseModel):
    host: str
    reachable: bool
    error: str | None = None
    role: str = "off"
    node_id: str = ""
    ready: bool = False
    active: bool = False
    checks: list[ClusterCheck] = []
    nodes: list[str] = []
    fresh: bool | None = None
    consensus: dict[str, Any] | None = None  # its /api/consensus/status, if any


class ClusterPaths(BaseModel):
    """Folders a cluster shares. None means Matricxon's default; no files are moved."""

    models_path: str | None = None
    prompt_cache_path: str | None = None
    # How other nodes reach the seed; needed before nodes join as voters.
    advertise_url: str | None = None
    # The node that starts the cluster: None keeps it, "" clears it.
    first_node: str | None = None


class NodeCheck(BaseModel):
    """What a candidate node reports before it is added."""

    address: str
    reachable: bool
    error: str | None = None
    fresh: bool = False
    in_cluster: bool = False  # already in this cluster: remove it first
    reasons: list[str] = []


class NodeAddress(BaseModel):
    address: str


class NodeAdd(BaseModel):
    address: str
    advertise_url: str | None = None  # how the OTHER nodes reach it, if different
    node_kind: Literal["compute", "voter"] = "compute"  # "voter": consensus only, never given chats


class VoterId(BaseModel):
    id: str


class QuorumState(BaseModel):
    quorum: str  # "off", "ok" or "lost"


class ClusterCache(BaseModel):
    """Saved-chat-cache limits, the same on every node."""

    enabled: bool
    budget_mb: int = Field(ge=0, le=1_048_576)
    ttl_hours: int = Field(ge=1, le=8760)


class ClusterTimers(BaseModel):
    """Consensus timers, the same on every node: election and heartbeat."""

    election_seconds: float
    heartbeat_seconds: float

    @model_validator(mode="after")
    def _make_sense(self) -> "ClusterTimers":
        if not 0.5 <= self.election_seconds <= 60:
            raise ValueError("election timeout must be between 0.5 and 60 seconds")
        if not 0.05 <= self.heartbeat_seconds <= 5:
            raise ValueError("heartbeat must be between 0.05 and 5 seconds")
        if self.heartbeat_seconds * 3 > self.election_seconds:
            raise ValueError("election timeout must be at least three heartbeats")
        return self


class ClusterState(BaseModel):
    """`enabled` is the saved choice; a differing host `role` means a restart is needed (`restart_needed`)."""

    enabled: bool
    supported: bool
    restart_needed: bool
    paths: ClusterPaths
    app_checks: list[ClusterCheck]
    cluster_id: str | None = None  # made by the cluster on first start
    quorum: str = "off"  # "off", "ok", or "lost" (no majority leader: changes paused)
    message: str | None = None  # what the last change could not do
    node_hosts: list[str] = []  # joiners added here (removable)
    timers: ClusterTimers | None = None  # None while consensus is off
    hosts: list[ClusterHost]
    admin_auth: bool = False  # pAIring signs its admin calls to the nodes (an admin secret is set)


class JoinCluster(BaseModel):
    """Join an existing cluster by any node's address and the cluster id."""

    address: str
    cluster_id: str = Field(min_length=8, max_length=64)
    admin_secret: str = Field(default="", max_length=200)  # needed when the cluster's nodes have one


class ClusterUpdate(BaseModel):
    """Turn cluster mode on or off."""

    enabled: bool


class MapNode(BaseModel):
    """One box on the Nodes map."""

    id: str
    kind: str  # "app" | "matricxon" | "storage" | "image" | "voter" (unmatched voter)
    name: str
    subtitle: str = ""
    status: str
    badges: list[str] = []
    address: str | None = None
    activatable: bool = False
    removable: bool = False
    demotable: bool = False  # voter that may step down (not the last)
    address_mismatch: str | None = None  # voter advertising a different address than recorded
    action_notes: dict[str, str] = {}  # why an action is unavailable
    details: dict[str, Any] = {}
    checks: list[ClusterCheck] = []


class PeerAddress(BaseModel):
    """One entry of a node's address book."""

    id: str
    address: str
    voter: bool
    this_node: bool = False
    seed: bool = False
    kind: str = "old"  # "voter" | "learner" | "old" (remembered only)
    old_ids: int = 0  # older identities at this address


class PeerList(BaseModel):
    """A page of one node's address book."""

    node: str
    advertise_url: str | None = None
    total: int
    offset: int
    limit: int
    peers: list[PeerAddress]


class MapLink(BaseModel):
    source: str
    target: str
    kind: str
    active: bool = True
    leader: bool = False


class ClusterMap(BaseModel):
    quorum: str = "off"
    enabled: bool
    supported: bool
    nodes: list[MapNode]
    links: list[MapLink]
    outside: list[str] = []  # listed hosts not in the cluster: not drawn


class ClusterResetResult(BaseModel):
    reset: list[str]
    failed: list[str]


class AdminSecret(BaseModel):
    secret: str = Field(default="", max_length=200)  # "" turns signing off
