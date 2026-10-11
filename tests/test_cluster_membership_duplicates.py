from app.services.cluster_membership import ClusterMembership

A, A2, B = "http://127.0.0.1:8420", "http://172.20.10.14:8420", "http://127.0.0.1:8441"


def _status(node_id):
    return {"node_id": node_id}


def test_the_same_node_under_two_addresses_keeps_one_listing():
    statuses = {A: _status("a"), A2: _status("a"), B: _status("b")}
    assert ClusterMembership._duplicates([A, B, A2], statuses, None) == [A2]


def test_the_chosen_first_node_is_the_one_kept():
    statuses = {A: _status("a"), A2: _status("a")}
    assert ClusterMembership._duplicates([A, A2], statuses, A2) == [A]


def test_silent_hosts_and_distinct_nodes_are_left_alone():
    statuses = {A: _status("a"), B: _status("b")}
    assert ClusterMembership._duplicates([A, B, A2], statuses, None) == []
