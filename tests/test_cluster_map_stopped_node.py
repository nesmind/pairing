from app.schemas.cluster import ClusterHost
from app.services.cluster_map_consensus import ConsensusLinks

A, B = "http://127.0.0.1:8420", "http://127.0.0.1:8441"


def _status(node_id, voters, members):
    role = "leader" if node_id == "a" else "follower"
    return {"node_id": node_id, "epoch": 3, "role": role, "voters": voters, "members": members}


def test_a_stopped_node_is_not_shown_twice():
    members = {"a": "http://172.20.10.14:8420", "b": "http://172.20.10.14:8441"}
    both = {A: _status("a", ["a", "b"], members), B: _status("b", ["a", "b"], members)}
    hosts = [ClusterHost(host=A, reachable=True), ClusterHost(host=B, reachable=True)]
    ghosts, _ = ConsensusLinks().of(hosts, both)
    assert ghosts == []  # running: matched by node id

    hosts[0] = ClusterHost(host=A, reachable=False)
    known = {A: {"node_id": "a", "advertise_url": "http://172.20.10.14:8420"}}
    ghosts, links = ConsensusLinks().of(hosts, {B: both[B]}, known)
    assert ghosts == []  # stopped: still its own (down) box, not a second "voter" box
    assert len(links) == 1 and not links[0].active


def test_a_stopped_node_is_matched_by_its_advertised_address_too():
    members = {"a": "http://172.20.10.14:8420", "b": "http://172.20.10.14:8441"}
    hosts = [ClusterHost(host=A, reachable=False), ClusterHost(host=B, reachable=True)]
    known = {A: {"node_id": "old-id", "advertise_url": "http://172.20.10.14:8420/"}}
    ghosts, _ = ConsensusLinks().of(hosts, {B: _status("b", ["a", "b"], members)}, known)
    assert ghosts == []


def test_an_unknown_voter_is_still_a_ghost():
    hosts = [ClusterHost(host=B, reachable=True)]
    members = {"b": "http://172.20.10.14:8441", "x": "http://10.9.9.9:8420"}
    ghosts, _ = ConsensusLinks().of(hosts, {B: _status("b", ["b", "x"], members)})
    assert [g.id for g in ghosts] == ["voter:x"]
