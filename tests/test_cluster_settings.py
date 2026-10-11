"""Cluster-wide settings go through any voter (not the seed); the map shows roles from consensus."""

import json

import httpx
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.routers import cluster_admin
from app.schemas.cluster import ClusterCache, ClusterTimers
from app.services import engine_service, settings_service
from app.services.cluster_map import ClusterMapBuilder
from app.services.cluster_service import ClusterService
from app.services.cluster_settings import ClusterSettings, ConsensusOffError

SEED, JOINER = "http://seed:8420", "http://joiner:8420"
TIMERS = {"election_seconds": 1.0, "heartbeat_seconds": 0.2}


def _status(node: str, role: str, voter: bool = True, quorum: bool = True) -> dict:
    return {
        "enabled": True,
        "node_id": node,
        "role": role,
        "is_voter": voter,
        "has_quorum": quorum,
        "epoch": 1,
        "applied_slot": 3,
        "voters": ["s", "j"],
        "members": {},
        "holds": [],
        "timers": TIMERS,
    }


def _cluster(statuses: dict[str, dict | None], puts: list):
    def handler(request: httpx.Request) -> httpx.Response:
        host = f"{request.url.scheme}://{request.url.host}:{request.url.port}"
        if statuses.get(host) is None:
            raise httpx.ConnectError("down")
        if request.url.path == "/api/consensus/status":
            return httpx.Response(200, json=statuses[host])
        if request.url.path == "/api/consensus/config":
            puts.append((host, json.loads(request.content)))
            return httpx.Response(200, json={"applied": True, "result": True})
        if request.url.path == "/api/cluster/readiness":
            body = {"role": "seed", "node_id": "x", "ready": True, "active": True, "checks": [], "nodes": []}
            return httpx.Response(200, json=body)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


@pytest.fixture(autouse=True)
async def _engine(db):
    await engine_service.set_active_engine(db, "matricxon")


@pytest.mark.asyncio
async def test_the_leader_is_the_entry_point_and_the_seed_may_be_gone(db):
    both = {SEED: _status("s", "follower"), JOINER: _status("j", "leader")}
    entry = await ClusterSettings(db, _cluster(both, []), [SEED, JOINER]).entry_voter()
    assert entry[0] == JOINER
    seed_down = {SEED: None, JOINER: _status("j", "follower")}
    entry = await ClusterSettings(db, _cluster(seed_down, []), [SEED, JOINER]).entry_voter()
    assert entry[0] == JOINER
    learner = {SEED: _status("s", "follower", voter=False)}
    assert await ClusterSettings(db, _cluster(learner, []), [SEED]).entry_voter() is None


@pytest.mark.asyncio
async def test_folders_go_through_the_cluster_when_it_runs_consensus(db, monkeypatch):
    puts: list = []
    pushed: list = []

    async def push(self, config):
        pushed.append(config)

    monkeypatch.setattr(ClusterService, "_push_to_seed", push)
    config = await settings_service.get_matricxon_server_config(db)
    config.models_path, config.prompt_cache_path = "/mnt/m", "/mnt/c"
    await settings_service.set_matricxon_server_config(db, config)
    transport = _cluster({SEED: _status("s", "leader"), JOINER: _status("j", "follower")}, puts)
    used, problem = await ClusterSettings(db, transport, [SEED, JOINER]).publish_paths("/mnt/m", "/mnt/c")
    assert used and problem is None
    assert puts == [(SEED, {"key": "cluster/paths", "value": {"models_dir": "/mnt/m", "prompt_cache_dir": "/mnt/c"}})]
    assert not pushed


@pytest.mark.asyncio
async def test_without_consensus_folders_are_not_published(db):
    used, problem = await ClusterSettings(db, _cluster({SEED: {"enabled": False}}, []), [SEED]).publish_paths("a", "b")
    assert (used, problem) == (False, None)


@pytest.mark.asyncio
async def test_timers_are_published_and_shown_in_the_state(db):
    puts: list = []
    transport = _cluster({SEED: _status("s", "leader")}, puts)
    new = ClusterTimers(election_seconds=5, heartbeat_seconds=1)
    assert await ClusterSettings(db, transport, [SEED]).set_timers(new) is None
    assert puts[0][1] == {"key": "cluster/timers", "value": {"election_seconds": 5.0, "heartbeat_seconds": 1.0}}
    state = await ClusterService(db, [SEED], transport).state()
    assert state.timers == ClusterTimers(**TIMERS)


def test_timers_must_make_sense():
    for bad in ((0.1, 0.05), (100, 1), (1, 0.5), (2, 0.01)):
        with pytest.raises(ValidationError):
            ClusterTimers(election_seconds=bad[0], heartbeat_seconds=bad[1])


@pytest.mark.asyncio
async def test_the_timers_endpoint_needs_consensus_and_a_quorum(db, monkeypatch):
    body = ClusterTimers(election_seconds=5, heartbeat_seconds=1)
    monkeypatch.setattr(ClusterService, "quorum_of", staticmethod(lambda hosts: "lost"))
    with pytest.raises(HTTPException) as lost:
        await cluster_admin.set_timers(body, db, None)
    assert lost.value.status_code == 409

    monkeypatch.setattr(ClusterService, "quorum_of", staticmethod(lambda hosts: "off"))

    async def nobody(self, timers):
        raise ConsensusOffError("no voter")

    monkeypatch.setattr(ClusterSettings, "set_timers", nobody)
    with pytest.raises(HTTPException) as off:
        await cluster_admin.set_timers(body, db, None)
    assert off.value.status_code == 409 and "consensus" in off.value.detail


@pytest.mark.asyncio
async def test_the_map_names_roles_by_consensus_not_by_who_started_the_cluster(db):
    config = await settings_service.get_matricxon_server_config(db)
    config.cluster_hosts, config.cluster_role = [JOINER], "seed"
    await settings_service.set_matricxon_server_config(db, config)
    statuses = {SEED: _status("s", "follower"), JOINER: _status("j", "leader")}
    transport = _cluster(statuses, [])
    nodes = {
        n.id: n
        for n in (await ClusterMapBuilder(db, ClusterService(db, [SEED, JOINER], transport), transport).build()).nodes
    }
    for host in (SEED, JOINER):
        assert "seed" not in nodes[host].badges and "joiner" not in nodes[host].badges
        assert "voter" in nodes[host].badges
    assert "leader" in nodes[JOINER].badges and "leader" not in nodes[SEED].badges


@pytest.mark.asyncio
async def test_the_quorum_endpoint_says_lost_when_no_voter_has_a_majority(db, monkeypatch):
    async def quorum(statuses: dict, hosts: list[str]) -> str:
        seen = await ClusterSettings(db, _cluster(statuses, []), hosts).voters_seen()
        return ClusterService.quorum_of(seen)

    split = {SEED: _status("s", "follower", quorum=False), JOINER: _status("j", "candidate", quorum=False)}
    assert await quorum(split, [SEED, JOINER]) == "lost"
    healthy = {SEED: _status("s", "leader"), JOINER: _status("j", "follower")}
    assert await quorum(healthy, [SEED, JOINER]) == "ok"
    assert await quorum({SEED: {"enabled": False}}, [SEED]) == "off"

    lost = await ClusterSettings(db, _cluster(split, []), [SEED, JOINER]).voters_seen()

    async def seen_lost(self):
        return lost

    monkeypatch.setattr(ClusterSettings, "voters_seen", seen_lost)
    assert (await cluster_admin.get_quorum(db, None)).quorum == "lost"


@pytest.mark.asyncio
async def test_cache_limits_are_published_through_the_cluster(db, monkeypatch):
    puts: list = []
    transport = _cluster({SEED: _status("s", "leader")}, puts)
    cache = ClusterCache(enabled=True, budget_mb=512, ttl_hours=24)
    assert await ClusterSettings(db, transport, [SEED]).set_cache(cache) is None
    assert puts == [(SEED, {"key": "cluster/cache", "value": {"enabled": True, "budget_mb": 512, "ttl_hours": 24}})]


def test_cache_limits_must_make_sense():
    for bad in (
        {"budget_mb": -1, "ttl_hours": 1},
        {"budget_mb": 1, "ttl_hours": 0},
        {"budget_mb": 1, "ttl_hours": 9999},
    ):
        with pytest.raises(ValidationError):
            ClusterCache(enabled=True, **bad)


@pytest.mark.asyncio
async def test_the_cache_endpoint_needs_a_quorum_and_falls_back_to_each_node_without_consensus(db, monkeypatch):
    from app.schemas import MatricxonCachePersistence
    from app.services.matricxon_cache_persistence import MatricxonCachePersistenceService

    body = ClusterCache(enabled=False, budget_mb=0, ttl_hours=1)
    monkeypatch.setattr(ClusterService, "quorum_of", staticmethod(lambda hosts: "lost"))
    with pytest.raises(HTTPException) as lost:
        await cluster_admin.set_cache(body, db, None)
    assert lost.value.status_code == 409

    monkeypatch.setattr(ClusterService, "quorum_of", staticmethod(lambda hosts: "off"))
    told: list = []
    done = MatricxonCachePersistence(enabled=False, budget_mb=0, ttl_hours=1, files=0, used_bytes=0, hosts=1)

    async def direct(self, change):
        told.append(change)
        return done

    monkeypatch.setattr(MatricxonCachePersistenceService, "update", direct)
    assert await cluster_admin.set_cache(body, db, None) == done
    assert told[0].budget_mb == 0 and told[0].enabled is False
