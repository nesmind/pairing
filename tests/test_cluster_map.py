"""The Nodes map: reports become boxes (status, badges) and links."""

import httpx
import pytest

from app.routers import cluster_admin
from app.schemas import SdCppConfig
from app.schemas.cluster import ClusterMap
from app.services import engine_service, image_engine_service, sdcpp_pool, settings_service
from app.services.cluster_map import ClusterMapBuilder
from app.services.cluster_service import ClusterService

A, B, C = "http://a:8420", "http://b:8420", "http://c:8420"


def _readiness(ready: bool = True, active: bool = True, role: str = "seed") -> dict:
    ok = "ok" if ready else "action_needed"
    checks = [
        {"id": "models", "status": ok, "what": "models", "why": "w"},
        {"id": "prompt_cache", "status": ok, "what": "cache", "why": "w"},
        {"id": "cache_key", "status": "ok", "what": "key", "why": "w"},
    ]
    return {"role": role, "node_id": "id", "ready": ready, "active": active, "checks": checks, "nodes": ["x"]}


def _consensus(node_id: str, role: str, voters: list[str], members: dict[str, str] | None = None) -> dict:
    return {
        "enabled": True,
        "node_id": node_id,
        "epoch": 2,
        "role": role,
        "leader": "mx1",
        "is_voter": True,
        "voters": voters,
        "applied_slot": 5,
        "config": {},
        "members": members or {},
    }


def _transport(readiness: dict, consensus: dict | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        host = f"{request.url.scheme}://{request.url.host}:{request.url.port}"
        if host not in readiness:
            raise httpx.ConnectError("down")
        if request.url.path == "/api/cluster/readiness":
            return httpx.Response(200, json=readiness[host])
        if request.url.path == "/api/consensus/status":
            if consensus and host in consensus:
                return httpx.Response(200, json=consensus[host])
            return httpx.Response(200, json={"enabled": False})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


@pytest.fixture(autouse=True)
async def _engine(db, monkeypatch):
    await engine_service.set_active_engine(db, "matricxon")
    config = await settings_service.get_matricxon_server_config(db)
    config.cluster_role = "seed"
    await settings_service.set_matricxon_server_config(db, config)

    async def nothing_running(_host: str) -> bool:
        return False

    monkeypatch.setattr(sdcpp_pool, "check_host", nothing_running)
    monkeypatch.setattr(sdcpp_pool, "get_effective_hosts", lambda: ["http://localhost:1234"])


async def _added(db, *hosts: str) -> None:
    """Hosts added from the Cluster page are drawn even before consensus."""
    config = await settings_service.get_matricxon_server_config(db)
    config.cluster_hosts = list(hosts)
    await settings_service.set_matricxon_server_config(db, config)


async def _build(db, hosts: list[str], readiness: dict, consensus: dict | None = None) -> ClusterMap:
    transport = _transport(readiness, consensus)
    return await ClusterMapBuilder(db, ClusterService(db, hosts, transport), transport).build()


def _by_id(result: ClusterMap) -> dict:
    return {n.id: n for n in result.nodes}


@pytest.mark.asyncio
async def test_a_lone_seed_is_pAIring_a_matricxon_node_and_the_storage(db):
    result = await _build(db, [A], {A: _readiness()})
    nodes = _by_id(result)
    assert set(nodes) == {"app", A, "storage"}
    assert nodes[A].kind == "matricxon" and nodes[A].status == "up" and nodes[A].badges == ["seed", "active"]
    assert nodes["storage"].status == "up" and nodes["app"].kind == "app"
    assert {(link.source, link.target, link.kind) for link in result.links} == {
        ("app", A, "api"),
        (A, "storage", "storage"),
    }


@pytest.mark.asyncio
async def test_status_follows_what_each_node_reports(db):
    config = await settings_service.get_matricxon_server_config(db)
    config.cluster_hosts = [B, C]
    await settings_service.set_matricxon_server_config(db, config)
    readiness = {A: _readiness(), B: _readiness(ready=False, active=False, role="joiner")}
    nodes = _by_id(await _build(db, [A, B, C], readiness))
    assert (nodes[A].status, nodes[B].status, nodes[C].status) == ("up", "warn", "down")
    assert nodes[B].badges == ["joiner"] and nodes[B].removable and not nodes[A].removable
    assert nodes[C].badges == [] and nodes[C].details["Error"]
    assert nodes[B].activatable is False
    assert nodes["storage"].status == "warn"
    assert [c.id for c in nodes["storage"].checks] == ["models", "prompt_cache", "cache_key"]


@pytest.mark.asyncio
async def test_a_ready_node_that_is_not_active_can_be_activated(db):
    nodes = _by_id(await _build(db, [A], {A: _readiness(ready=True, active=False)}))
    assert nodes[A].activatable and "active" not in nodes[A].badges


@pytest.mark.asyncio
async def test_voters_and_the_leader_are_marked_and_linked_in_a_star(db):
    readiness = {A: _readiness(), B: _readiness(active=True, role="joiner")}
    consensus = {
        A: _consensus("mx1", "leader", ["mx1", "mx2", "mx3"], {"mx1": A, "mx2": B, "mx3": C}),
        B: _consensus("mx2", "follower", ["mx1", "mx2", "mx3"], {"mx1": A, "mx2": B, "mx3": C}),
    }
    result = await _build(db, [A, B], readiness, consensus)
    nodes = _by_id(result)
    assert "leader" in nodes[A].badges and "voter" in nodes[A].badges
    assert "voter" in nodes[B].badges and "leader" not in nodes[B].badges
    assert "epoch 2" in nodes[A].details["Consensus"]
    assert nodes["voter:mx3"].kind == "voter" and nodes["voter:mx3"].status == "warn"
    links = {(link.source, link.target): link for link in result.links if link.kind == "consensus"}
    assert set(links) == {(A, B), (A, "voter:mx3")} and all(link.leader for link in links.values())
    assert links[(A, B)].active and not links[(A, "voter:mx3")].active


@pytest.mark.asyncio
async def test_without_a_leader_the_voters_are_linked_pairwise(db):
    readiness = {A: _readiness(), B: _readiness()}
    members = {"mx1": A, "mx2": B}
    consensus = {
        A: _consensus("mx1", "candidate", ["mx1", "mx2"], members),
        B: _consensus("mx2", "follower", ["mx1", "mx2"], members),
    }
    result = await _build(db, [A, B], readiness, consensus)
    pairs = [(link.source, link.target, link.leader) for link in result.links if link.kind == "consensus"]
    assert pairs == [(A, B, False)]


@pytest.mark.asyncio
async def test_the_image_engine_is_not_part_of_the_map(db):
    await image_engine_service.set_sdcpp_config(db, SdCppConfig(binary_path="/opt/sd-server"))
    result = await _build(db, [A], {A: _readiness()})
    assert {n.kind for n in result.nodes} == {"app", "matricxon", "storage"}


@pytest.mark.asyncio
async def test_pAIrings_own_requirements_show_on_its_node(db, monkeypatch):
    from app.services.cluster_service import ClusterService as Service

    class Probe:
        def is_shared(self, _path) -> bool:
            return False

        def fstype(self, _path) -> str:
            return "ext4"

    transport = _transport({A: _readiness()})
    result = await ClusterMapBuilder(db, Service(db, [A], transport, Probe()), transport).build()
    app = _by_id(result)["app"]
    assert app.status == "warn" and app.checks[0].id == "images"


@pytest.mark.asyncio
async def test_other_engines_have_no_map(db):
    await engine_service.set_active_engine(db, "ollama")
    result = await _build(db, [A], {A: _readiness()})
    assert result.supported is False and result.nodes == [] and result.links == []


@pytest.mark.asyncio
async def test_the_route_returns_the_map(db, monkeypatch):
    result = await cluster_admin.get_map(db=db, _admin=None)
    assert result.supported is True and {n.kind for n in result.nodes} >= {"app"}


@pytest.mark.asyncio
async def test_the_jobs_a_node_owns_show_as_badges(db):
    members = {"mx1": A}
    status = _consensus("mx1", "leader", ["mx1"], members) | {"holds": ["model_pull", "cache_clean"]}
    nodes = _by_id(await _build(db, [A], {A: _readiness()}, {A: status}))
    assert {"leader", "voter", "pulls", "cleans"} <= set(nodes[A].badges)
    quiet = _consensus("mx1", "follower", ["mx1"], members) | {"holds": ["cache_clean"]}
    nodes = _by_id(await _build(db, [A], {A: _readiness()}, {A: quiet}))
    assert "cleans" in nodes[A].badges and "pulls" not in nodes[A].badges


# big clusters


def _many(count: int) -> list[str]:
    return [f"http://n{i}:8420" for i in range(count)]


def _slow_transport(hosts: list[str], delay: float = 0.05, seen: list | None = None):
    import asyncio

    async def handler(request: httpx.Request) -> httpx.Response:
        host = f"{request.url.scheme}://{request.url.host}:{request.url.port}"
        if seen is not None:
            seen.append((host, request.url.path))
        await asyncio.sleep(delay)
        if request.url.path == "/api/cluster/readiness":
            return httpx.Response(200, json=_readiness())
        return httpx.Response(200, json={"enabled": False})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_sixty_nodes_are_read_at_the_same_time_not_one_after_another(db):
    import time

    hosts = _many(60)
    transport = _slow_transport(hosts, delay=0.05)  # 60 x 2 calls x 50 ms = 6 s serially
    started = time.monotonic()
    result = await ClusterMapBuilder(db, ClusterService(db, hosts, transport), transport).build()
    assert time.monotonic() - started < 1.5
    assert len([n for n in result.nodes if n.kind == "matricxon"]) == 60


@pytest.mark.asyncio
async def test_a_compact_map_drops_details_and_checklists(db):
    members = {"mx1": A, "mx2": B}
    consensus = {
        A: _consensus("mx1", "leader", ["mx1", "mx2"], members),
        B: _consensus("mx2", "follower", ["mx1", "mx2"], members),
    }
    transport = _transport({A: _readiness(), B: _readiness()}, consensus)
    builder = ClusterMapBuilder(db, ClusterService(db, [A, B], transport), transport)
    full, compact = await builder.build(), await builder.build(compact=True)
    assert any(n.details and n.checks for n in full.nodes if n.kind == "matricxon")
    assert all(not n.details and not n.checks for n in compact.nodes)
    assert {n.id for n in compact.nodes} == {n.id for n in full.nodes}
    assert {link.kind for link in compact.links} == {"api", "consensus", "storage"}
    assert {n.name for n in compact.nodes if n.kind == "matricxon"} == {"Matricxon"}
    leader = next(n for n in compact.nodes if n.id == A)
    assert {"voter", "leader"} <= set(leader.badges) and leader.status == "up"


@pytest.mark.asyncio
async def test_one_nodes_full_record_costs_two_calls_not_the_whole_cluster(db):
    hosts = _many(40)
    seen: list = []
    transport = _slow_transport(hosts, delay=0, seen=seen)
    builder = ClusterMapBuilder(db, ClusterService(db, hosts, transport), transport)
    from app.services import matricxon_pool

    monkey = pytest.MonkeyPatch()
    monkey.setattr(matricxon_pool, "get_effective_hosts", lambda: hosts)
    try:
        node = await builder.node(hosts[7])
    finally:
        monkey.undo()
    assert node.id == hosts[7] and node.details["Address"] == hosts[7] and node.checks
    assert {host for host, _ in seen} == {hosts[7]}
    assert len(seen) == 2


@pytest.mark.asyncio
async def test_the_node_route_returns_a_record_or_404(db, monkeypatch):
    from fastapi import HTTPException

    async def none(self, node_id):
        return None

    monkeypatch.setattr(ClusterMapBuilder, "node", none)
    with pytest.raises(HTTPException) as missing:
        await cluster_admin.get_map_node(id="nope", db=db, _admin=None)
    assert missing.value.status_code == 404
    result = await cluster_admin.get_map(compact=True, db=db, _admin=None)
    assert result.supported is True


# quorum


def _voter(node_id: str, role: str, has_quorum: bool = True, voter: bool = True) -> dict:
    return _consensus(node_id, role, ["mx1", "mx2"], {"mx1": A, "mx2": B}) | {
        "has_quorum": has_quorum,
        "is_voter": voter,
    }


async def _quorum_of(db, statuses: dict) -> tuple[str, ClusterMap]:
    transport = _transport({A: _readiness(), B: _readiness()}, statuses)
    service = ClusterService(db, [A, B], transport)
    return (await service.state()).quorum, await ClusterMapBuilder(db, service, transport).build()


@pytest.mark.asyncio
async def test_quorum_is_off_without_consensus_and_ok_with_one_leader_behind_a_majority(db):
    assert (await _quorum_of(db, {}))[0] == "off"
    ok, result = await _quorum_of(db, {A: _voter("mx1", "leader"), B: _voter("mx2", "follower")})
    assert ok == "ok" and result.quorum == "ok"
    assert not any("no quorum" in n.badges for n in result.nodes)


@pytest.mark.asyncio
async def test_quorum_is_lost_with_no_leader_or_a_leader_without_a_majority(db):
    followers = {A: _voter("mx1", "follower", False), B: _voter("mx2", "candidate", False)}
    state, result = await _quorum_of(db, followers)
    assert state == "lost" and result.quorum == "lost"
    assert {"no quorum"} <= set(_by_id(result)[A].badges)
    assert _by_id(result)[A].status == "warn"

    lonely_leader = {A: _voter("mx1", "leader", has_quorum=False), B: _voter("mx2", "follower", False)}
    assert (await _quorum_of(db, lonely_leader))[0] == "lost"  # leading is not enough: needs a majority
    two_leaders = {A: _voter("mx1", "leader"), B: _voter("mx2", "leader")}
    assert (await _quorum_of(db, two_leaders))[0] == "lost"


@pytest.mark.asyncio
async def test_a_learner_that_is_not_a_voter_yet_is_not_flagged_for_having_no_quorum(db):
    statuses = {A: _voter("mx1", "leader"), B: _voter("mx2", "follower", has_quorum=False, voter=False)}
    state, result = await _quorum_of(db, statuses)
    assert state == "ok" and "no quorum" not in _by_id(result)[B].badges and _by_id(result)[B].status == "up"


@pytest.mark.asyncio
async def test_each_node_is_read_once_for_readiness_and_once_for_consensus_not_twice_per_page(db):
    seen: list = []
    hosts = _many(5)
    transport = _slow_transport(hosts, delay=0, seen=seen)
    builder = ClusterMapBuilder(db, ClusterService(db, hosts, transport), transport)
    await builder.build(compact=True)
    per_host = {h: [p for host, p in seen if host == h] for h in hosts}
    assert all(sorted(paths) == ["/api/cluster/readiness", "/api/consensus/status"] for paths in per_host.values())


@pytest.mark.asyncio
async def test_a_big_compact_map_keeps_only_the_voter_links(db):
    hosts = _many(8)
    transport = _slow_transport(hosts, delay=0)
    compact = await ClusterMapBuilder(db, ClusterService(db, hosts, transport), transport).build(compact=True)
    assert {link.kind for link in compact.links} <= {"consensus"}


@pytest.mark.asyncio
async def test_any_node_can_be_removed_once_the_cluster_runs_consensus_except_the_last_voter(db):
    members = {"mx1": A, "mx2": B}
    two_voters = {
        A: _consensus("mx1", "leader", ["mx1", "mx2"], members),
        B: _consensus("mx2", "follower", ["mx1", "mx2"], members),
    }
    nodes = _by_id(await _build(db, [A, B], {A: _readiness(), B: _readiness()}, two_voters))
    assert nodes[A].removable and nodes[B].removable

    one_voter = {
        A: _consensus("mx1", "leader", ["mx1"], {"mx1": A}),
        B: {**_consensus("mx2", "follower", ["mx1"]), "is_voter": False},
    }
    nodes = _by_id(await _build(db, [A, B], {A: _readiness(), B: _readiness()}, one_voter))
    assert not nodes[A].removable  # only voter: make another first
    assert "only voter" in nodes[A].action_notes["remove"]
    assert nodes[B].removable

    alone = _by_id(await _build(db, [A], {A: _readiness()}, {A: _consensus("mx1", "leader", ["mx1"])}))
    assert not alone[A].removable
    no_consensus = _by_id(await _build(db, [A, B], {A: _readiness(), B: _readiness()}))
    assert not no_consensus[A].removable


@pytest.mark.asyncio
async def test_a_node_that_runs_consensus_but_does_not_vote_can_still_be_activated(db):
    """Active (announced) is not a voter: the button stays for a learner."""
    members = {"mx1": A}
    states = {
        A: _consensus("mx1", "leader", ["mx1"], members),
        B: {**_consensus("mx2", "follower", ["mx1"]), "is_voter": False},
    }
    nodes = _by_id(await _build(db, [A, B], {A: _readiness(), B: _readiness()}, states))
    assert nodes[B].activatable and "learner" in nodes[B].badges
    assert not nodes[A].activatable


@pytest.mark.asyncio
async def test_the_panels_own_record_of_a_node_agrees_with_the_map_about_removing_it(db):
    """Bug: the panel reads one node alone yet must know the cluster's size."""
    members = {"mx1": A, "mx2": B}
    states = {
        A: _consensus("mx1", "follower", ["mx1", "mx2"], members),
        B: _consensus("mx2", "leader", ["mx1", "mx2"], members),
    }
    transport = _transport({A: _readiness(), B: _readiness()}, states)
    builder = ClusterMapBuilder(db, ClusterService(db, [A, B], transport), transport)

    from app.services import matricxon_pool

    monkey = pytest.MonkeyPatch()
    monkey.setattr(matricxon_pool, "get_effective_hosts", lambda: [A, B])
    try:
        one = await builder.node(A)
    finally:
        monkey.undo()
    assert one is not None and one.removable


@pytest.mark.asyncio
async def test_only_a_voter_that_is_not_the_last_one_can_be_made_a_learner(db):
    members = {"mx1": A, "mx2": B}
    KEEP3 = {"target_voters": 3}
    two = {
        A: _consensus("mx1", "leader", ["mx1", "mx2"], members) | KEEP3,
        B: _consensus("mx2", "follower", ["mx1", "mx2"], members) | KEEP3,
    }
    nodes = _by_id(await _build(db, [A, B], {A: _readiness(), B: _readiness()}, two))
    # cluster keeps 3 voters itself: with 2 a node cannot be a learner
    assert not nodes[A].demotable and "keeps 3 voters" in nodes[A].action_notes["demote"]
    four = {"mx1", "mx2", "mx3", "mx4"}
    many = {
        A: _consensus("mx1", "leader", sorted(four), members) | KEEP3,
        B: _consensus("mx2", "follower", sorted(four), members) | KEEP3,
    }
    nodes = _by_id(await _build(db, [A, B], {A: _readiness(), B: _readiness()}, many))
    assert nodes[A].demotable and nodes[B].demotable

    one = {
        A: _consensus("mx1", "leader", ["mx1"], {"mx1": A}),
        B: {**_consensus("mx2", "follower", ["mx1"]), "is_voter": False},
    }
    nodes = _by_id(await _build(db, [A, B], {A: _readiness(), B: _readiness()}, one))
    assert not nodes[A].demotable
    assert not nodes[B].demotable


@pytest.mark.asyncio
async def test_an_added_node_is_not_removable_when_it_is_the_only_voter(db):
    """Adding from the Cluster page must not bypass the last-voter rule (bug)."""
    config = await settings_service.get_matricxon_server_config(db)
    config.cluster_hosts = [B]
    await settings_service.set_matricxon_server_config(db, config)
    states = {
        A: {**_consensus("mx1", "follower", ["mx2"]), "is_voter": False},
        B: _consensus("mx2", "leader", ["mx2"], {"mx2": B}),
    }
    nodes = _by_id(await _build(db, [A, B], {A: _readiness(), B: _readiness()}, states))
    assert not nodes[B].removable and "only voter" in nodes[B].action_notes["remove"]
    assert nodes[A].removable


@pytest.mark.asyncio
async def test_every_unavailable_action_says_why(db):
    """The panel greys out buttons with reasons instead of hiding them."""
    await _added(db, B, C)
    members = {"mx1": A}
    states = {A: _consensus("mx1", "leader", ["mx1"], members)}
    nodes = _by_id(await _build(db, [A, B], {A: _readiness(), B: _readiness(ready=False)}, states))
    a = nodes[A].action_notes
    assert not nodes[A].demotable and "only voter" in a["demote"] and "only voter" in a["remove"]
    assert "already active" in a["activate"]

    down = _by_id(await _build(db, [A, C], {A: _readiness()}, {A: _consensus("mx1", "leader", ["mx1"])}))
    assert "does not answer" in down[C].action_notes["activate"]

    await _added(db)
    unready = _by_id(await _build(db, [A, B], {A: _readiness(), B: _readiness(ready=False)}))
    assert "checklist" in unready[B].action_notes["activate"]
    assert "does not run consensus" in unready[B].action_notes["remove"]
    assert "does not run consensus" in unready[B].action_notes["demote"]


@pytest.mark.asyncio
async def test_the_cluster_role_shown_for_a_node_is_its_consensus_role_not_its_setup_label(db):
    """Once consensus runs the panel shows the node's consensus role, not its .env label (old seed, now a learner, still
    said "seed")."""
    members = {"mx2": B}
    states = {
        A: {**_consensus("mx1", "follower", ["mx2"]), "is_voter": False},
        B: _consensus("mx2", "leader", ["mx2"], members),
    }
    nodes = _by_id(await _build(db, [A, B], {A: _readiness(role="seed"), B: _readiness(role="joiner")}, states))
    assert nodes[A].details["Cluster role"] == "learner"
    assert nodes[B].details["Cluster role"] == "leader"

    voter = {A: _consensus("mx1", "follower", ["mx1", "mx2"]), B: _consensus("mx2", "leader", ["mx1", "mx2"])}
    nodes = _by_id(await _build(db, [A, B], {A: _readiness(), B: _readiness()}, voter))
    assert nodes[A].details["Cluster role"] == "voter"

    before = _by_id(await _build(db, [A], {A: _readiness(role="seed")}))
    assert before[A].details["Cluster role"] == "seed"


@pytest.mark.asyncio
async def test_the_node_panel_shows_only_the_advertised_address_the_long_list_is_a_button(db):
    members = {"mx1": A, "mx2": B}
    status = _consensus("mx1", "leader", ["mx1", "mx2"], members)
    status |= {"advertise_url": "http://host.containers.internal:8420", "addresses": members}
    nodes = _by_id(await _build(db, [A, B], {A: _readiness(), B: _readiness()}, {A: status}))
    assert nodes[A].details["Advertises"] == "http://host.containers.internal:8420"
    assert "Peer addresses" not in nodes[A].details and "Start-up seeds" not in nodes[A].details


@pytest.mark.asyncio
async def test_a_voter_that_advertises_another_address_than_the_cluster_has_is_flagged(db):
    members = {"mx1": "http://lab:8420", "mx2": B}
    moved = _consensus("mx1", "leader", ["mx1", "mx2"], members) | {"advertise_url": "http://real:8420"}
    same = _consensus("mx2", "follower", ["mx1", "mx2"], members) | {"advertise_url": B}
    nodes = _by_id(await _build(db, [A, B], {A: _readiness(), B: _readiness()}, {A: moved, B: same}))
    assert "http://real:8420" in nodes[A].address_mismatch and "http://lab:8420" in nodes[A].address_mismatch
    assert "Address mismatch" in nodes[A].details
    assert nodes[B].address_mismatch is None and "Address mismatch" not in nodes[B].details


async def test_a_host_the_running_cluster_does_not_include_is_marked_not_in_cluster(db):
    a, b = "http://a:8420", "http://b:8420"
    await _added(db, b)
    ready = _readiness()
    cluster = _consensus("n1", "leader", ["n1"])

    def handler(request):
        if request.url.path == "/api/consensus/status":
            return httpx.Response(200, json=cluster if request.url.host == "a" else {"enabled": False})
        return httpx.Response(200, json=ready)

    transport = httpx.MockTransport(handler)
    result = await ClusterMapBuilder(db, ClusterService(db, [a, b], transport), transport).build()
    nodes = _by_id(result)
    assert nodes[b].badges == ["not in cluster"] and "voter" in nodes[a].badges


@pytest.mark.asyncio
async def test_a_follower_that_hears_a_leader_pairing_does_not_list_means_a_quorum(db):
    """The leader may have left pAIring's host list: only followers are seen."""
    only_followers = {A: _voter("mx2", "follower"), B: _voter("mx3", "follower")}
    assert (await _quorum_of(db, only_followers))[0] == "ok"
    deaf = {A: _voter("mx2", "follower", has_quorum=False), B: _voter("mx3", "follower", has_quorum=False)}
    assert (await _quorum_of(db, deaf))[0] == "lost"


@pytest.mark.asyncio
async def test_a_voter_pairing_has_no_node_for_can_be_removed_unless_it_is_the_only_one(db):
    members = {"mx1": A, "mx2": "http://gone:8420"}
    two = {A: _consensus("mx1", "leader", ["mx1", "mx2"], members)}
    nodes = _by_id(await _build(db, [A], {A: _readiness()}, two))
    assert nodes["voter:mx2"].removable
    one = {A: _consensus("mx1", "leader", ["mx1"], {"mx1": A})}
    assert not [n for n in (await _build(db, [A], {A: _readiness()}, one)).nodes if n.kind == "voter"]


@pytest.mark.asyncio
async def test_a_listed_node_the_cluster_does_not_have_can_be_added_from_its_panel(db):
    await _added(db, B)
    members = {"mx1": A}
    statuses = {A: _consensus("mx1", "leader", ["mx1"], members)}
    nodes = _by_id(await _build(db, [A, B], {A: _readiness(), B: _readiness()}, statuses))
    assert nodes[B].badges == ["not in cluster"] and nodes[B].activatable


@pytest.mark.asyncio
async def test_a_learner_replaying_an_old_epoch_does_not_bring_back_voters_that_left(db):
    leader = _consensus("mx1", "leader", ["mx1"], {"mx1": A}) | {"epoch": 8, "applied_total": 500}
    stale = _consensus("mx2", "follower", ["mx2", "old"], {"mx2": B, "old": "http://gone:8420"}) | {
        "epoch": 3,
        "is_voter": False,
        "applied_total": 100,
    }
    result = await _build(db, [A, B], {A: _readiness(), B: _readiness()}, {A: leader, B: stale})
    assert not [n for n in result.nodes if n.kind == "voter"]


@pytest.mark.asyncio
async def test_a_learner_behind_the_log_is_amber_with_its_progress_and_ok_once_it_caught_up(db):
    leader = _consensus("mx1", "leader", ["mx1"], {"mx1": A}) | {"applied_total": 1000}

    async def learner(done: int):
        mine = _consensus("mx2", "follower", ["mx1"], {"mx1": A}) | {"is_voter": False, "applied_total": done}
        return _by_id(await _build(db, [A, B], {A: _readiness(), B: _readiness()}, {A: leader, B: mine}))[B]

    behind = await learner(250)
    assert behind.status == "warn" and "replaying the log 25%" in behind.badges
    assert (await learner(980)).status == "up"


@pytest.mark.asyncio
async def test_a_host_that_is_only_listed_is_not_drawn_while_the_cluster_grows(db):
    """After a reset old hosts stay listed but are not drawn while the cluster grows."""
    members = {"mx1": A}
    statuses = {A: _consensus("mx1", "leader", ["mx1"], members)}
    result = await _build(db, [A, B, C], {A: _readiness(), B: _readiness(), C: _readiness()}, statuses)
    assert {n.id for n in result.nodes if n.kind == "matricxon"} == {A} and result.outside == [B, C]
    await _added(db, B)
    again = await _build(db, [A, B, C], {A: _readiness(), B: _readiness(), C: _readiness()}, statuses)
    assert {n.id for n in again.nodes if n.kind == "matricxon"} == {A, B} and again.outside == [C]


@pytest.mark.asyncio
async def test_while_the_first_node_is_set_up_only_it_is_drawn(db):
    config = await settings_service.get_matricxon_server_config(db)
    config.first_node = A
    await settings_service.set_matricxon_server_config(db, config)
    result = await _build(db, [A, B], {A: _readiness(), B: _readiness()})
    assert {n.id for n in result.nodes if n.kind == "matricxon"} == {A} and result.outside == [B]


@pytest.mark.asyncio
async def test_with_cluster_mode_off_no_host_is_drawn_after_a_reset(db):
    config = await settings_service.get_matricxon_server_config(db)
    config.cluster_role = "off"
    await settings_service.set_matricxon_server_config(db, config)
    result = await _build(db, [A, B], {A: _readiness(), B: _readiness()})
    assert not [n for n in result.nodes if n.kind == "matricxon"] and result.outside == [A, B]


@pytest.mark.asyncio
async def test_storage_is_green_when_no_node_is_drawn_and_ignores_hosts_outside_the_cluster(db):
    config = await settings_service.get_matricxon_server_config(db)
    config.cluster_role = "off"
    await settings_service.set_matricxon_server_config(db, config)
    off = _by_id(await _build(db, [A], {A: _readiness(ready=False)}))
    assert off["storage"].status == "up"
    await _added(db)
    config.cluster_role, config.first_node = "seed", A
    await settings_service.set_matricxon_server_config(db, config)
    nodes = _by_id(await _build(db, [A, B], {A: _readiness(), B: _readiness(ready=False)}))
    assert nodes["storage"].status == "up" and B not in nodes
