"""Cluster page > add/check/remove a node: mocked candidate, saved config and host pool."""

import json

import httpx
import pytest
from fastapi import HTTPException

from app.routers import cluster_admin
from app.schemas.cluster import NodeAdd, NodeAddress
from app.services import cluster_nodes, matricxon_pool, server_pool_broadcast, settings_service
from app.services.cluster_nodes import ClusterNodes, NodeNotAllowedError, NodeNotFreshError

NODE = "http://10.0.0.7:8420"


@pytest.fixture(autouse=True)
def _quorum_ok(monkeypatch):
    from app.services.cluster_service import ClusterService

    monkeypatch.setattr(ClusterService, "quorum_of", staticmethod(lambda hosts: "ok"))


@pytest.fixture(autouse=True)
def _quiet_broadcast(monkeypatch):
    async def none(_path):
        return []

    monkeypatch.setattr(server_pool_broadcast, "broadcast_refresh", none)
    yield
    matricxon_pool.refresh_from_config(
        __import__("app.schemas", fromlist=["MatricxonServerConfig"]).MatricxonServerConfig()
    )


async def _join(db, *hosts: str) -> None:
    """Puts `hosts` in the saved hosts AND the live pool."""
    config = await settings_service.get_matricxon_server_config(db)
    config.cluster_hosts = list(hosts)
    await settings_service.set_matricxon_server_config(db, config)
    matricxon_pool.refresh_from_config(config)


async def _configure(db, role="seed", models="/mnt/m", cache="/mnt/c") -> None:
    config = await settings_service.get_matricxon_server_config(db)
    config.cluster_role, config.models_path, config.prompt_cache_path = role, models, cache
    await settings_service.set_matricxon_server_config(db, config)


def _node(fresh=True, provision_status=200, seen=None):
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append((request.method, request.url.path, request.content))
        if request.url.path == "/api/cluster/fresh":
            return httpx.Response(200, json={"fresh": fresh, "reasons": [] if fresh else ["2 installed model(s)"]})
        return httpx.Response(provision_status, json={"restarting": True, "detail": "boom"})

    return httpx.MockTransport(handler)


def test_addresses_are_normalized():
    assert ClusterNodes.normalize(" 10.0.0.7:8420/ ") == NODE
    assert ClusterNodes.normalize("https://n.example") == "https://n.example"
    with pytest.raises(NodeNotAllowedError):
        ClusterNodes.normalize("ftp://x")


@pytest.mark.asyncio
async def test_check_reports_fresh_populated_unreachable_and_old_nodes(db):
    assert (await ClusterNodes(db, _node()).check(NODE)).fresh
    populated = await ClusterNodes(db, _node(fresh=False)).check(NODE)
    assert populated.reachable and not populated.fresh and "installed model" in populated.reasons[0]

    def down(request):
        raise httpx.ConnectError("refused")

    assert (await ClusterNodes(db, httpx.MockTransport(down)).check(NODE)).error == "refused"
    old = await ClusterNodes(db, httpx.MockTransport(lambda r: httpx.Response(404))).check(NODE)
    assert "no cluster mode" in old.error


@pytest.mark.asyncio
async def test_add_pushes_the_shared_folders_and_joins_the_pool(db):
    await _configure(db)
    seen: list = []
    await ClusterNodes(db, _node(seen=seen)).add(NODE)

    post = next(s for s in seen if s[0] == "POST")
    assert json.loads(post[2]) == {"models_dir": "/mnt/m", "prompt_cache_dir": "/mnt/c", "restart": True}
    assert (await settings_service.get_matricxon_server_config(db)).cluster_hosts == [NODE]
    assert NODE in matricxon_pool.get_effective_hosts()


@pytest.mark.asyncio
async def test_add_refuses_a_populated_node_and_changes_nothing(db):
    await _configure(db)
    seen: list = []
    with pytest.raises(NodeNotFreshError, match="installed model"):
        await ClusterNodes(db, _node(fresh=False, seen=seen)).add(NODE)
    assert all(s[0] == "GET" for s in seen)
    assert (await settings_service.get_matricxon_server_config(db)).cluster_hosts == []


@pytest.mark.asyncio
async def test_add_needs_cluster_mode_and_the_shared_folders(db):
    await _configure(db, role="off")
    with pytest.raises(NodeNotAllowedError, match="mode on first"):
        await ClusterNodes(db, _node()).add(NODE)
    await _configure(db, cache=None)
    with pytest.raises(NodeNotAllowedError, match="folders first"):
        await ClusterNodes(db, _node()).add(NODE)


@pytest.mark.asyncio
async def test_add_refuses_the_seed_itself_and_duplicates(db):
    await _configure(db)
    with pytest.raises(NodeNotAllowedError, match="already a node"):
        await ClusterNodes(db, _node()).add(matricxon_pool.LOCAL_MATRICXON_HOST)
    await ClusterNodes(db, _node()).add(NODE)
    with pytest.raises(NodeNotAllowedError, match="already a node"):
        await ClusterNodes(db, _node()).add(NODE)


@pytest.mark.asyncio
async def test_a_node_that_refuses_the_configuration_is_not_added(db):
    await _configure(db)
    with pytest.raises(NodeNotAllowedError, match="refused"):
        await ClusterNodes(db, _node(provision_status=400)).add(NODE)
    with pytest.raises(NodeNotFreshError):
        await ClusterNodes(db, _node(provision_status=409)).add(NODE)
    assert (await settings_service.get_matricxon_server_config(db)).cluster_hosts == []


@pytest.mark.asyncio
async def test_remove_takes_the_node_out_of_the_pool(db):
    await _configure(db)
    await ClusterNodes(db, _node()).add(NODE)
    await ClusterNodes(db).remove(NODE)
    assert NODE not in matricxon_pool.get_effective_hosts()
    with pytest.raises(NodeNotAllowedError, match="not a node of this cluster"):
        await ClusterNodes(db).remove(NODE)


@pytest.mark.asyncio
async def test_router_maps_errors_to_400_and_409(db, monkeypatch):
    await _configure(db, role="off")
    with pytest.raises(HTTPException) as bad:
        await cluster_admin.add_node(NodeAdd(address=NODE), db=db, _admin=None)
    assert bad.value.status_code == 400

    async def not_fresh(self, address, advertise_url=None, node_kind="compute"):
        raise NodeNotFreshError("populated")

    monkeypatch.setattr(cluster_nodes.ClusterNodes, "add", not_fresh)
    with pytest.raises(HTTPException) as conflict:
        await cluster_admin.add_node(NodeAdd(address=NODE), db=db, _admin=None)
    assert conflict.value.status_code == 409


# joiners become voters

SEED = "http://localhost:8420"


def _consensus_status(node_id: str, voters: list[str], members: dict | None = None) -> dict:
    return {
        "enabled": True,
        "node_id": node_id,
        "target_voters": 3,
        "voters": voters,
        "members": members or {},
        "is_voter": node_id in voters,
        "role": "leader",
        "has_quorum": True,
    }


def _cluster_node(seed_status: dict | None, joiner_status: dict | None = None, seen: list | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        host = f"{request.url.scheme}://{request.url.host}:{request.url.port}"
        if seen is not None:
            seen.append((request.method, host, request.url.path, request.content))
        if request.url.path == "/api/cluster/fresh":
            return httpx.Response(200, json={"fresh": True, "reasons": []})
        if request.url.path == "/api/cluster/readiness":
            return httpx.Response(200, json={"node_id": "joiner"})
        if request.url.path == "/api/consensus/reach":
            return httpx.Response(200, json={"reachable": True, "node_id": "joiner" if host == SEED else "voter"})
        if request.url.path == "/api/consensus/status":
            status = seed_status if host == SEED else joiner_status
            return httpx.Response(200, json=status or {"enabled": False})
        return httpx.Response(200, json={"restarting": True, "ok": True})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_a_joiner_is_told_its_address_and_the_voters_when_the_seed_runs_consensus(db):
    await _configure(db)
    seen: list = []
    seed = _consensus_status("mx1", ["mx1"], {"mx1": SEED})
    await ClusterNodes(db, _cluster_node(seed, seen=seen)).add(NODE)
    provision = next(s for s in seen if s[2] == "/api/cluster/provision")
    assert json.loads(provision[3]) == {
        "models_dir": "/mnt/m",
        "prompt_cache_dir": "/mnt/c",
        "restart": True,
        "advertise_url": NODE,
        "consensus_seeds": {"mx1": SEED},
    }


@pytest.mark.asyncio
async def test_without_consensus_on_the_seed_the_joiner_gets_only_the_folders(db):
    await _configure(db)
    seen: list = []
    await ClusterNodes(db, _cluster_node(None, seen=seen)).add(NODE)
    body = json.loads(next(s for s in seen if s[2] == "/api/cluster/provision")[3])
    assert body == {"models_dir": "/mnt/m", "prompt_cache_dir": "/mnt/c", "restart": True}


@pytest.mark.asyncio
async def test_activating_makes_a_restarted_joiner_a_voter_once(db):
    await _configure(db)
    await _join(db, NODE)
    seen: list = []
    transport = _cluster_node(_consensus_status("mx1", ["mx1"]), _consensus_status("j1", ["j1"]), seen)
    await ClusterNodes(db, transport).ensure_voters()
    posts = [s for s in seen if s[0] == "POST"]
    assert [(p[1], p[2], json.loads(p[3])) for p in posts] == [
        (SEED, "/api/consensus/voters", {"id": "j1", "address": NODE})
    ]

    seen.clear()
    transport = _cluster_node(_consensus_status("mx1", ["mx1", "j1"]), _consensus_status("j1", ["mx1", "j1"]), seen)
    await ClusterNodes(db, transport).ensure_voters()
    assert not [s for s in seen if s[0] == "POST"]


@pytest.mark.asyncio
async def test_a_joiner_that_has_not_restarted_into_consensus_is_left_for_the_next_activate(db):
    await _configure(db)
    await _join(db, NODE)
    seen: list = []
    await ClusterNodes(db, _cluster_node(_consensus_status("mx1", ["mx1"]), None, seen)).ensure_voters()
    assert not [s for s in seen if s[0] == "POST"]
    await ClusterNodes(db, _cluster_node(None, None, seen)).ensure_voters()


@pytest.mark.asyncio
async def test_removing_a_node_also_removes_it_as_a_voter(db):
    await _configure(db)
    await _join(db, NODE)
    seen: list = []
    transport = _cluster_node(_consensus_status("mx1", ["mx1", "j1"]), _consensus_status("j1", ["mx1", "j1"]), seen)
    await ClusterNodes(db, transport).remove(NODE)
    deletes = [s for s in seen if s[0] == "DELETE"]
    assert [(d[1], d[2]) for d in deletes] == [(SEED, "/api/consensus/voters/j1")]
    assert (await settings_service.get_matricxon_server_config(db)).cluster_hosts == []


@pytest.mark.asyncio
async def test_membership_changes_are_refused_without_a_quorum(db, monkeypatch):
    from app.services.cluster_service import ClusterService

    monkeypatch.setattr(ClusterService, "quorum_of", staticmethod(lambda hosts: "lost"))
    for call in (
        cluster_admin.add_node(NodeAdd(address=NODE), db, None),
        cluster_admin.remove_node(NodeAddress(address=NODE), db, None),
        cluster_admin.activate_cluster(db, None),
    ):
        with pytest.raises(HTTPException) as caught:
            await call
        assert caught.value.status_code == 409 and "quorum" in caught.value.detail.lower()


@pytest.mark.asyncio
async def test_a_removed_node_can_be_reset_to_fresh(db):
    seen: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        return httpx.Response(200, json={"restarting": True})

    await ClusterNodes(db, httpx.MockTransport(handler)).reset(NODE)
    assert seen == [("GET", "/api/consensus/status"), ("POST", "/api/cluster/reset")]


@pytest.mark.asyncio
async def test_a_node_still_in_the_pool_is_not_reset(db):
    await _configure(db)
    await _join(db, NODE)
    seen: list = []
    transport = httpx.MockTransport(lambda r: seen.append(r) or httpx.Response(200, json={}))
    with pytest.raises(NodeNotAllowedError, match="still a node of this cluster"):
        await ClusterNodes(db, transport).reset(NODE)
    assert seen == []


@pytest.mark.asyncio
async def test_the_nodes_own_refusal_is_passed_on(db):
    refusal = httpx.Response(409, json={"detail": "still a voter of its cluster: remove it first"})
    transport = httpx.MockTransport(lambda r: refusal)
    with pytest.raises(cluster_nodes.NodeResetRefusedError, match="still a voter"):
        await ClusterNodes(db, transport).reset(NODE)
    with pytest.raises(HTTPException) as caught:
        monkey = pytest.MonkeyPatch()
        monkey.setattr(ClusterNodes, "reset", _refuse)
        try:
            await cluster_admin.reset_node(NodeAddress(address=NODE), db, None)
        finally:
            monkey.undo()
    assert caught.value.status_code == 409


async def _refuse(self, address):
    raise cluster_nodes.NodeResetRefusedError("no")


# first node is removed like any other once consensus runs


def _status(node_id: str, voters: list[str]) -> dict:
    return _consensus_status(node_id, voters, {v: v for v in voters})


@pytest.mark.asyncio
async def test_the_first_node_can_be_removed_and_pairing_switches_to_remote_mode(db):
    await _configure(db)
    await _join(db, NODE)
    seen: list = []
    transport = _cluster_node(_status("mx1", ["mx1", "j1"]), _status("j1", ["mx1", "j1"]), seen)

    await ClusterNodes(db, transport).remove(SEED)

    assert [(d[1], d[2]) for d in seen if d[0] == "DELETE"] == [(SEED, "/api/consensus/voters/mx1")]
    config = await settings_service.get_matricxon_server_config(db)
    assert (config.mode, config.remote_hosts, config.cluster_hosts) == ("remote", [NODE], [])
    assert matricxon_pool.get_effective_hosts() == [NODE]


@pytest.mark.asyncio
async def test_the_first_node_is_not_removed_without_consensus_nor_as_the_last_node_or_voter(db):
    await _configure(db)
    transport = _cluster_node(None, None)
    with pytest.raises(NodeNotAllowedError, match="needs the cluster to be running"):
        await ClusterNodes(db, transport).remove(SEED)

    transport = _cluster_node(_status("mx1", ["mx1"]), None)
    with pytest.raises(NodeNotAllowedError, match="the only node"):
        await ClusterNodes(db, transport).remove(SEED)

    await _join(db, NODE)
    transport = _cluster_node(_status("mx1", ["mx1"]), _status("j1", ["mx1"]))
    with pytest.raises(NodeNotAllowedError, match="only voter"):
        await ClusterNodes(db, transport).remove(SEED)
    config = await settings_service.get_matricxon_server_config(db)
    assert config.mode == "local"


@pytest.mark.asyncio
async def test_activating_also_promotes_nodes_that_are_not_in_the_added_list(db):
    await _configure(db)
    config = await settings_service.get_matricxon_server_config(db)
    config.mode, config.remote_hosts, config.cluster_hosts = "remote", [SEED, NODE], []
    await settings_service.set_matricxon_server_config(db, config)
    matricxon_pool.refresh_from_config(config)
    seen: list = []
    transport = _cluster_node(_status("mx1", ["mx1"]), _status("j1", ["j1"]), seen)
    await ClusterNodes(db, transport).ensure_voters()
    assert [(p[1], json.loads(p[3])["id"]) for p in seen if p[0] == "POST"] == [(SEED, "j1")]


@pytest.mark.asyncio
async def test_checking_a_node_that_is_still_in_the_cluster_says_remove_it_first(db):
    await _configure(db)
    await _join(db, NODE)
    seen: list = []
    transport = httpx.MockTransport(
        lambda r: seen.append(r) or httpx.Response(200, json={"fresh": False, "reasons": ["x"]})
    )
    for address in (NODE, "http://localhost:8420", "http://127.0.0.1:8420", "127.0.0.1:8420/"):
        result = await ClusterNodes(db, transport).check(address)
        assert result.in_cluster and result.reachable and not result.fresh
        assert "remove it first" in result.reasons[0]
    assert not [r for r in seen if r.url.path == "/api/cluster/fresh"]

    result = await ClusterNodes(db, transport).check("http://10.9.9.9:8420")
    assert not result.in_cluster and result.reasons == ["x"]
    assert len([r for r in seen if r.url.path == "/api/cluster/fresh"]) == 1


@pytest.mark.asyncio
async def test_the_same_machine_under_another_name_is_the_same_node(db):
    await _configure(db)
    await _join(db, NODE)
    assert ClusterNodes.pool_spelling("http://127.0.0.1:8420") == matricxon_pool.LOCAL_MATRICXON_HOST
    assert ClusterNodes.pool_spelling("http://127.0.0.1:9999") is None
    with pytest.raises(NodeNotAllowedError, match="already a node of this cluster"):
        await ClusterNodes(db, _node()).add("http://127.0.0.1:8420")
    seen: list = []
    transport = _cluster_node(_status("mx1", ["mx1", "j1"]), _status("j1", ["mx1", "j1"]), seen)
    await ClusterNodes(db, transport).remove("http://127.0.0.1:8420")
    assert [(d[1], d[2]) for d in seen if d[0] == "DELETE"] == [(SEED, "/api/consensus/voters/mx1")]


def _at(status: dict, epoch: int, slot: int) -> dict:
    return {**status, "epoch": epoch, "applied_slot": slot}


@pytest.mark.asyncio
async def test_a_learner_that_is_far_behind_is_not_made_a_voter_until_it_has_caught_up(db):
    """Promoting a far-behind node costs the quorum while it catches up."""
    await _configure(db)
    await _join(db, NODE)
    leader = _at(_status("mx1", ["mx1"]), 7, 2000)

    for behind in (_at(_status("j1", ["mx1"]), 5, 800), _at(_status("j1", ["mx1"]), 7, 1500)):
        seen: list = []
        message = await ClusterNodes(db, _cluster_node(leader, behind, seen)).ensure_voters()
        assert not [s for s in seen if s[0] == "POST"]
        assert message and NODE in message and "catching up" in message

    seen = []
    close = _at(_status("j1", ["mx1"]), 7, 1980)
    assert await ClusterNodes(db, _cluster_node(leader, close, seen)).ensure_voters() is None
    assert [json.loads(s[3])["id"] for s in seen if s[0] == "POST"] == ["j1"]


# a voter can stop voting and stay as a learner


def _voting_cluster(voters: list[str], seen: list, ok: bool = True):
    """Both nodes answer status; the voter change goes to the first (the leader)."""

    def handler(request: httpx.Request) -> httpx.Response:
        host = f"{request.url.scheme}://{request.url.host}:{request.url.port}"
        seen.append((request.method, host, request.url.path))
        if request.url.path == "/api/consensus/status":
            node = "mx1" if host == SEED else "j1"
            return httpx.Response(200, json=_status(node, voters))
        if request.method == "DELETE":
            return httpx.Response(200, json={"ok": ok})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_a_voter_can_be_demoted_to_a_learner_through_the_leader(db):
    await _configure(db)
    await _join(db, NODE)
    seen: list = []
    await ClusterNodes(db, _voting_cluster(["mx1", "j1", "x1", "x2"], seen)).demote(NODE)
    assert ("DELETE", SEED, "/api/consensus/voters/j1") in seen
    # the node stays in the pool
    assert NODE in matricxon_pool.get_effective_hosts()


@pytest.mark.asyncio
async def test_the_last_voter_a_learner_and_strangers_cannot_be_demoted(db):
    await _configure(db)
    await _join(db, NODE)
    seen: list = []
    with pytest.raises(NodeNotAllowedError, match="only voter"):
        await ClusterNodes(db, _voting_cluster(["mx1"], seen)).demote(SEED)
    with pytest.raises(NodeNotAllowedError, match="not a voter"):
        await ClusterNodes(db, _voting_cluster(["mx1"], seen)).demote(NODE)
    with pytest.raises(NodeNotAllowedError, match="keeps 3 voters"):
        await ClusterNodes(db, _voting_cluster(["mx1", "j1", "x1"], seen)).demote(NODE)
    with pytest.raises(NodeNotAllowedError, match="not a node of this cluster"):
        await ClusterNodes(db, _voting_cluster(["mx1", "j1"], seen)).demote("http://10.9.9.9:8420")
    assert not [s for s in seen if s[0] == "DELETE"]


@pytest.mark.asyncio
async def test_a_demotion_the_cluster_did_not_make_is_reported_and_the_router_maps_it(db, monkeypatch):
    await _configure(db)
    await _join(db, NODE)
    with pytest.raises(NodeNotAllowedError, match="did not make the change"):
        await ClusterNodes(db, _voting_cluster(["mx1", "j1", "x1", "x2"], [], ok=False)).demote(NODE)

    from app.services.cluster_service import ClusterService

    monkeypatch.setattr(ClusterService, "quorum_of", staticmethod(lambda hosts: "lost"))
    with pytest.raises(HTTPException) as no_quorum:
        await cluster_admin.demote_node(NodeAddress(address=NODE), db, None)
    assert no_quorum.value.status_code == 409
    monkeypatch.setattr(ClusterService, "quorum_of", staticmethod(lambda hosts: "ok"))
    with pytest.raises(HTTPException) as refused:
        await cluster_admin.demote_node(NodeAddress(address="http://10.9.9.9:8420"), db, None)
    assert refused.value.status_code == 400


@pytest.mark.asyncio
async def test_a_node_is_made_a_voter_under_the_address_it_advertises_not_pairings_name_for_it(db):
    """Bug: the old first node was promoted as http://localhost:8420, which is the joiner itself from
    its view."""
    await _configure(db)
    await _join(db, NODE)
    advertised = "http://host.containers.internal:8420"
    leader = _status("j1", ["j1"])

    def handler(request: httpx.Request) -> httpx.Response:
        host = f"{request.url.scheme}://{request.url.host}:{request.url.port}"
        if request.url.path == "/api/consensus/status":
            if host == NODE:
                return httpx.Response(200, json=leader)
            return httpx.Response(200, json={**_status("mx1", ["j1"]), "is_voter": False, "advertise_url": advertised})
        posts.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": True})

    posts: list = []
    await ClusterNodes(db, httpx.MockTransport(handler)).ensure_voters()
    assert posts == [{"id": "mx1", "address": advertised}]


@pytest.mark.asyncio
async def test_saving_the_matricxon_form_does_not_wipe_the_nodes_added_on_the_cluster_page(db):
    """Bug: every form Save emptied cluster_hosts, so an added node vanished."""
    from app.routers import matricxon_admin
    from app.schemas import MatricxonServerConfig

    await _configure(db)
    await _join(db, NODE)
    form = MatricxonServerConfig(mode="remote", remote_hosts=[SEED], cluster_role="seed")
    assert form.cluster_hosts == []
    await matricxon_admin.update_config(form, db=db, _admin=None)

    saved = await settings_service.get_matricxon_server_config(db)
    assert saved.cluster_hosts == [NODE] and saved.mode == "remote"
    assert matricxon_pool.get_effective_hosts() == [SEED, NODE]


@pytest.mark.asyncio
async def test_an_added_node_that_is_the_only_voter_cannot_be_removed_either(db):
    """Bug: Remove worked for an added node even when it was the only voter."""
    await _configure(db)
    await _join(db, NODE)
    seen: list = []
    only = _at(_status("j1", ["j1"]), 5, 100)
    learner = _at({**_status("mx1", ["j1"]), "is_voter": False}, 5, 100)

    with pytest.raises(NodeNotAllowedError, match="only voter"):
        await ClusterNodes(db, _cluster_node(learner, only, seen)).remove(NODE)

    config = await settings_service.get_matricxon_server_config(db)
    assert config.cluster_hosts == [NODE]
    assert not [s for s in seen if s[0] == "DELETE"]


@pytest.mark.asyncio
async def test_a_node_stays_in_pairings_list_when_the_cluster_does_not_take_it_out_of_the_voters(db):
    await _configure(db)
    await _join(db, NODE)
    refused = _voting_cluster(["mx1", "j1"], [], ok=False)
    with pytest.raises(NodeNotAllowedError, match="did not take"):
        await ClusterNodes(db, refused).remove(NODE)
    assert (await settings_service.get_matricxon_server_config(db)).cluster_hosts == [NODE]


@pytest.mark.asyncio
async def test_before_consensus_an_added_node_can_still_be_removed(db):
    await _configure(db)
    await _join(db, NODE)
    await ClusterNodes(db, _cluster_node(None, None)).remove(NODE)
    assert (await settings_service.get_matricxon_server_config(db)).cluster_hosts == []


@pytest.mark.asyncio
async def test_a_voter_that_moved_has_its_new_address_recorded_when_the_cluster_is_activated(db):
    """A node with a real address instead of a lab one: the voter list still has the old one."""
    await _configure(db)
    await _join(db, NODE)
    real = "http://10.0.0.5:8420"
    posts: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        host = f"{request.url.scheme}://{request.url.host}:{request.url.port}"
        if request.url.path == "/api/consensus/status":
            if host == NODE:
                return httpx.Response(
                    200, json={**_status("j1", ["j1", "mx1"]), "members": {"j1": NODE, "mx1": "http://lab:8420"}}
                )
            return httpx.Response(200, json={**_status("mx1", ["j1", "mx1"]), "advertise_url": real})
        posts.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": True})

    await ClusterNodes(db, httpx.MockTransport(handler)).ensure_voters()
    assert posts == [{"id": "mx1", "address": real}]


def test_an_address_counts_as_moved_only_when_both_are_known_and_differ():
    cluster = {"members": {"a": "http://x:1/"}}
    assert ClusterNodes.address_moved({"node_id": "a", "advertise_url": "http://y:1"}, cluster)
    assert not ClusterNodes.address_moved({"node_id": "a", "advertise_url": "http://x:1"}, cluster)
    assert not ClusterNodes.address_moved({"node_id": "a", "advertise_url": None}, cluster)
    assert not ClusterNodes.address_moved({"node_id": "b", "advertise_url": "http://y:1"}, cluster)


async def test_reset_cluster_resets_every_node_and_clears_the_cluster_settings(db):
    from app.services.cluster_reset import ClusterReset

    await _configure(db)
    await _join(db, NODE, "http://10.0.0.8:8420", "http://10.0.0.9:8420")
    config = await settings_service.get_matricxon_server_config(db)
    config.first_node = NODE
    await settings_service.set_matricxon_server_config(db, config)
    calls: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.url.host, request.url.params.get("force")))
        if request.url.host == "10.0.0.8":
            raise httpx.ConnectError("down")
        if request.url.host == "10.0.0.9":  # old Matricxon: no `force`, refuses as a voter
            return httpx.Response(409, json={"detail": "this node is still a voter of its cluster"})
        return httpx.Response(200, json={"restarting": True})

    result = await ClusterReset(db, httpx.MockTransport(handler)).run()
    config = await settings_service.get_matricxon_server_config(db)
    assert ("10.0.0.7", "true") in calls and sorted(result.failed) == ["http://10.0.0.8:8420", "http://10.0.0.9:8420"]
    assert config.cluster_role == "off" and config.cluster_hosts == [] and config.models_path is None
    assert config.first_node is None
    assert config.mode == "remote" and len(config.remote_hosts) == 4


async def test_a_plain_host_the_running_cluster_does_not_include_can_be_added(db, monkeypatch):
    other = "http://10.0.0.9:8420"
    await _configure(db)
    config = await settings_service.get_matricxon_server_config(db)
    config.mode, config.remote_hosts = "remote", [NODE, other]
    await settings_service.set_matricxon_server_config(db, config)
    matricxon_pool.refresh_from_config(config)

    async def running(_self, _config):
        return NODE, _status("j1", ["j1"])

    monkeypatch.setattr(ClusterNodes, "_seed_consensus", running)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/consensus/status":
            on = request.url.host == "10.0.0.7"
            return httpx.Response(200, json=_status("j1", ["j1"]) if on else {"enabled": False})
        if request.url.path == "/api/cluster/readiness":
            return httpx.Response(200, json={"node_id": "joiner"})
        if request.url.path == "/api/consensus/reach":
            return httpx.Response(
                200, json={"reachable": True, "node_id": "joiner" if request.url.host == "10.0.0.7" else "voter"}
            )
        return httpx.Response(200, json={"fresh": True, "reasons": [], "restarting": True})

    nodes = ClusterNodes(db, httpx.MockTransport(handler))
    assert (await nodes.check(other)).in_cluster is False
    assert await nodes.add(other) == other
    config = await settings_service.get_matricxon_server_config(db)
    assert config.cluster_hosts == [other] and config.remote_hosts == [NODE]
    with pytest.raises(NodeNotAllowedError):
        await nodes.add(NODE)


@pytest.mark.asyncio
async def test_a_voter_that_is_no_node_of_pairings_list_can_be_taken_out_of_the_voters(db):
    await _configure(db)
    seen: list = []
    nodes = ClusterNodes(db, _voting_cluster(["mx1", "ghost"], seen))
    await nodes.drop_unknown_voter("ghost")
    assert ("DELETE", SEED, "/api/consensus/voters/ghost") in seen

    with pytest.raises(NodeNotAllowedError, match="not a voter"):
        await ClusterNodes(db, _voting_cluster(["mx1"], [])).drop_unknown_voter("ghost")
    with pytest.raises(NodeNotAllowedError, match="only voter"):
        await ClusterNodes(db, _voting_cluster(["mx1"], [])).drop_unknown_voter("mx1")
    other: list = []
    with pytest.raises(NodeNotAllowedError, match="remove it as a node"):
        await ClusterNodes(db, _voting_cluster(["mx1", "j1"], other)).drop_unknown_voter("mx1")
    assert not [s for s in other if s[0] == "DELETE"]


@pytest.mark.asyncio
async def test_reset_to_fresh_first_takes_a_node_that_still_votes_out_of_the_voters(db):
    await _configure(db)
    ghost = "http://10.9.9.9:8420"
    calls: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        host = f"{request.url.scheme}://{request.url.host}:{request.url.port}"
        calls.append((request.method, host, request.url.path, request.url.params.get("force")))
        if request.url.path == "/api/consensus/status":
            node = "mx1" if host == SEED else "ghost"
            return httpx.Response(200, json=_status(node, ["mx1", "ghost"]))
        return httpx.Response(200, json={"ok": True, "restarting": True})

    await ClusterNodes(db, httpx.MockTransport(handler)).reset(ghost)
    deleted = [c for c in calls if c[0] == "DELETE"]
    assert deleted == [("DELETE", SEED, "/api/consensus/voters/ghost", None)]
    assert ("POST", ghost, "/api/cluster/reset", "true") in calls
    assert calls.index(deleted[0]) < calls.index(("POST", ghost, "/api/cluster/reset", "true"))


@pytest.mark.asyncio
async def test_removing_a_node_that_no_longer_answers_as_a_member_tells_the_cluster_to_forget_its_old_id(db):
    """a reset node got a new identity; its old id stayed, so pAIring re-added the address after every removal."""
    await _configure(db)
    await _join(db, NODE)
    calls: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        host = f"{request.url.scheme}://{request.url.host}:{request.url.port}"
        calls.append((request.method, host, request.url.path, request.url.params.get("leave")))
        if request.url.path == "/api/consensus/status":
            if host == SEED:
                return httpx.Response(200, json={**_status("mx1", ["mx1", "mx2"]), "nodes": {"mx1": SEED, "old": NODE}})
            return httpx.Response(200, json={"enabled": False})
        return httpx.Response(200, json={"ok": True})

    await ClusterNodes(db, httpx.MockTransport(handler)).remove(NODE)
    deleted = [c for c in calls if c[0] == "DELETE"]
    assert deleted == [("DELETE", SEED, "/api/consensus/voters/old", "true")]
    assert NODE not in matricxon_pool.get_effective_hosts()


def _reach_cluster(reach: dict, seen: list):
    """A running cluster whose reach check says `reach`; the candidate is fresh."""

    def handler(request: httpx.Request) -> httpx.Response:
        host = f"{request.url.scheme}://{request.url.host}:{request.url.port}"
        seen.append((request.method, host, request.url.path, request.content))
        if request.url.path == "/api/cluster/fresh":
            return httpx.Response(200, json={"fresh": True, "reasons": []})
        if request.url.path == "/api/cluster/readiness":
            return httpx.Response(200, json={"node_id": "joiner"})
        if request.url.path == "/api/consensus/reach":
            return httpx.Response(200, json=reach if host == SEED else {"reachable": True, "node_id": "voter"})
        if request.url.path == "/api/consensus/status":
            return httpx.Response(200, json=_status("mx1", ["mx1"]) if host == SEED else {"enabled": False})
        return httpx.Response(200, json={"restarting": True, "ok": True})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_a_node_the_cluster_cannot_reach_at_its_address_is_refused_before_anything_changes(db):
    """127.0.0.1:8420 from pAIring is the leader itself seen from a container; the new node never caught up."""
    await _configure(db)
    for reach in ({"reachable": False, "detail": "refused"}, {"reachable": True, "node_id": "the-leader"}):
        seen: list = []
        with pytest.raises(NodeNotAllowedError, match="OTHER nodes use"):
            await ClusterNodes(db, _reach_cluster(reach, seen)).add(NODE)
        assert not [s for s in seen if s[2] == "/api/cluster/provision"]


@pytest.mark.asyncio
async def test_the_address_the_other_nodes_use_is_what_the_node_advertises(db):
    await _configure(db)
    seen: list = []
    transport = _reach_cluster({"reachable": True, "node_id": "joiner"}, seen)
    await ClusterNodes(db, transport).add(NODE, "http://real.host:8420/")
    provision = next(s for s in seen if s[2] == "/api/cluster/provision")
    assert json.loads(provision[3])["advertise_url"] == "http://real.host:8420"
    reach = next(s for s in seen if s[2] == "/api/consensus/reach")
    assert json.loads(reach[3]) == {"address": "http://real.host:8420"}
    assert NODE in matricxon_pool.get_effective_hosts()


@pytest.mark.asyncio
async def test_the_cluster_reset_reports_each_node_as_it_finishes_and_why_one_needs_the_script(db):
    from app.services.cluster_reset import ClusterReset

    down, old, fine = "http://10.0.0.8:8420", "http://10.0.0.9:8420", NODE
    await _configure(db)
    await _join(db, fine, down, old)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "10.0.0.8":
            raise httpx.ConnectError("refused")
        if request.url.host == "10.0.0.9":
            return httpx.Response(404)
        return httpx.Response(200, json={"restarting": True})

    events = [e async for e in ClusterReset(db, httpx.MockTransport(handler)).stream()]
    assert events[0] == {"event": "start", "total": 4}
    nodes = {e["host"]: e for e in events if e["event"] == "node"}
    assert nodes[fine]["ok"] and not nodes[down]["ok"] and "not reachable" in nodes[down]["reason"]
    assert "no reset" in nodes[old]["reason"]
    assert events[-1]["event"] == "done" and sorted(events[-1]["failed"]) == sorted([down, old])
    assert (await _after_config(db)).cluster_role == "off"


async def _after_config(db):
    return await settings_service.get_matricxon_server_config(db)


@pytest.mark.asyncio
async def test_the_reset_stream_is_sent_as_server_sent_events(db, monkeypatch):
    class Session:
        async def __aenter__(self):
            return db

        async def __aexit__(self, *exc):
            return False

    class Stub:
        def __init__(self, _db):
            pass

        async def stream(self):
            yield {"event": "start", "total": 1}
            yield {"event": "done", "reset": ["a"], "failed": []}

    monkeypatch.setattr(cluster_admin, "AsyncSessionLocal", Session)
    monkeypatch.setattr(cluster_admin, "ClusterReset", Stub)
    response = await cluster_admin.reset_cluster_stream(_admin=None)
    chunks = [c async for c in response.body_iterator]
    assert response.media_type == "text/event-stream"
    assert chunks == [
        'data: {"event": "start", "total": 1}\n\n',
        'data: {"event": "done", "reset": ["a"], "failed": []}\n\n',
    ]


@pytest.mark.asyncio
async def test_a_new_node_that_cannot_reach_any_voter_is_refused_before_it_is_changed(db):
    await _configure(db)
    seen: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        host = f"{request.url.scheme}://{request.url.host}:{request.url.port}"
        seen.append(request.url.path)
        if request.url.path == "/api/cluster/fresh":
            return httpx.Response(200, json={"fresh": True, "reasons": []})
        if request.url.path == "/api/cluster/readiness":
            return httpx.Response(200, json={"node_id": "joiner"})
        if request.url.path == "/api/consensus/reach":
            return httpx.Response(
                200, json={"reachable": True, "node_id": "joiner"} if host == SEED else {"reachable": False}
            )
        if request.url.path == "/api/consensus/status":
            return httpx.Response(200, json=_status("mx1", ["mx1"]) if host == SEED else {"enabled": False})
        return httpx.Response(200, json={"restarting": True})

    with pytest.raises(NodeNotAllowedError, match="cannot reach the cluster's node"):
        await ClusterNodes(db, httpx.MockTransport(handler)).add(NODE)
    assert "/api/cluster/provision" not in seen
