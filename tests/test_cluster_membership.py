"""ClusterMembership: pAIring's node list follows the cluster's. Mock nodes only."""

import httpx
import pytest

from app.schemas import MatricxonServerConfig
from app.services import matricxon_pool, server_pool_broadcast, settings_service
from app.services.cluster_membership import ClusterMembership

A, B = "http://a:8420", "http://b:8420"


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    async def none(_path):
        return []

    monkeypatch.setattr(server_pool_broadcast, "broadcast_refresh", none)
    yield
    matricxon_pool.refresh_from_config(MatricxonServerConfig())


async def _listed(db, hosts: list[str], role: str = "seed", cluster: list[str] | None = None) -> None:
    config = await settings_service.get_matricxon_server_config(db)
    config.mode, config.remote_hosts, config.cluster_role = "remote", hosts, role
    config.cluster_hosts = cluster or []
    await settings_service.set_matricxon_server_config(db, config)
    matricxon_pool.refresh_from_config(config)


def _node(node_id: str, nodes: dict[str, str], removed: list[str] | None = None, slot: int = 10) -> dict:
    return {
        "enabled": True,
        "node_id": node_id,
        "epoch": 1,
        "applied_slot": slot,
        "nodes": nodes,
        "removed": removed or [],
    }


def _transport(statuses: dict[str, dict | None]):
    def handler(request: httpx.Request) -> httpx.Response:
        status = statuses.get(request.url.host)
        return httpx.Response(200, json=status or {"enabled": False})

    return httpx.MockTransport(handler)


async def _after(db) -> MatricxonServerConfig:
    return await settings_service.get_matricxon_server_config(db)


@pytest.mark.asyncio
async def test_a_node_the_cluster_has_that_pairing_does_not_list_is_added(db):
    await _listed(db, [A])
    nodes = {"n1": A, "n2": B}
    changed = await ClusterMembership().sync(db, _transport({"a": _node("n1", nodes)}))
    assert changed and (await _after(db)).cluster_hosts == [B]


@pytest.mark.asyncio
async def test_a_node_removed_on_purpose_is_dropped_from_the_list(db):
    await _listed(db, [A], cluster=[B])
    statuses = {"a": _node("n1", {"n1": A}, removed=["n2"]), "b": _node("n2", {"n1": A}, removed=["n2"], slot=5)}
    assert await ClusterMembership().sync(db, _transport(statuses))
    config = await _after(db)
    assert config.cluster_hosts == [] and config.remote_hosts == [A]


@pytest.mark.asyncio
async def test_removing_the_last_plain_host_keeps_the_added_nodes_instead_of_falling_back_to_local(db):
    await _listed(db, [A], cluster=[B])
    statuses = {"a": _node("n1", {"n2": B}, removed=["n1"]), "b": _node("n2", {"n2": B}, removed=["n1"], slot=11)}
    # n1 (listed as A) was removed: B is left and must not become "local mode"
    assert await ClusterMembership().sync(db, _transport(statuses))
    config = await _after(db)
    assert config.mode == "remote" and config.remote_hosts == [B]


@pytest.mark.asyncio
async def test_a_node_is_matched_by_its_id_so_pairings_own_spelling_is_kept(db):
    await _listed(db, [A])
    nodes = {"n1": "http://host.containers.internal:8420"}
    assert not await ClusterMembership().sync(db, _transport({"a": _node("n1", nodes)}))
    assert (await _after(db)).cluster_hosts == []


@pytest.mark.asyncio
async def test_nothing_changes_when_cluster_mode_is_off_nobody_answers_or_there_is_no_node_list(db):
    await _listed(db, [A], role="off")
    assert not await ClusterMembership().sync(db, _transport({"a": _node("n1", {"n1": A, "n2": B})}))
    await _listed(db, [A])
    assert not await ClusterMembership().sync(db, _transport({}))
    assert not await ClusterMembership().sync(db, _transport({"a": _node("n1", {})}))
    assert (await _after(db)).cluster_hosts == []


@pytest.mark.asyncio
async def test_a_listed_node_that_does_not_answer_is_kept(db):
    await _listed(db, [A, B])
    nodes = {"n1": A}
    assert not await ClusterMembership().sync(db, _transport({"a": _node("n1", nodes)}))
    assert (await _after(db)).remote_hosts == [A, B]


@pytest.mark.asyncio
async def test_a_restarting_node_is_not_listed_again_under_the_name_the_cluster_knows_it_by(db):
    await _listed(db, [A, B])
    named = "http://b-inside:8420"
    nodes = {"n1": A, "n2": named}
    membership = ClusterMembership()
    both = {"a": _node("n1", nodes), "b": _node("n2", nodes) | {"advertise_url": named}}
    assert not await membership.sync(db, _transport(both))
    assert not await membership.sync(db, _transport({"a": _node("n1", nodes)}))
    assert (await _after(db)).cluster_hosts == []
    assert await membership.sync(db, _transport({"a": _node("n1", nodes | {"n3": "http://c:8420"})})) is True
    assert (await _after(db)).cluster_hosts == ["http://c:8420"]


@pytest.mark.asyncio
async def test_a_second_listing_of_a_node_pairing_cannot_reach_by_the_clusters_name_is_dropped(db):
    cluster_name = "http://host.containers.internal:8420"
    await _listed(db, [A], cluster=[B, cluster_name])
    nodes = {"n1": A, "n2": cluster_name}
    statuses = {"a": _node("n1", nodes), "b": _node("n2", nodes, slot=9)}
    assert await ClusterMembership().sync(db, _transport(statuses))
    assert (await _after(db)).cluster_hosts == [B]
