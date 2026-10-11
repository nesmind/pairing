"""One node's address book in pages."""

import httpx
import pytest
from fastapi import HTTPException

from app.routers import cluster_admin
from app.schemas import MatricxonServerConfig
from app.services import engine_service, matricxon_pool, settings_service
from app.services.cluster_nodes import NodeNotAllowedError
from app.services.cluster_peers import MAX_PAGE, ClusterPeers

NODE = "http://10.0.0.7:8420"


@pytest.fixture(autouse=True)
async def _engine(db):
    await engine_service.set_active_engine(db, "matricxon")
    config = await settings_service.get_matricxon_server_config(db)
    config.cluster_hosts = [NODE]
    await settings_service.set_matricxon_server_config(db, config)
    matricxon_pool.refresh_from_config(config)
    yield
    matricxon_pool.refresh_from_config(MatricxonServerConfig())


def _status(addresses: dict[str, str], voters: list[str], me: str = "n00001", seeds: dict | None = None) -> dict:
    return {
        "enabled": True,
        "node_id": me,
        "voters": voters,
        "addresses": addresses,
        "seeds": seeds or {},
        "advertise_url": "http://adv:8420",
    }


def _peers(status: dict | None, seen: list | None = None) -> ClusterPeers:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(f"{request.url.host}{request.url.path}")
        if status is None:
            return httpx.Response(200, json={"enabled": False})
        return httpx.Response(200, json=status)

    return ClusterPeers(None, httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_the_list_starts_with_this_node_then_the_voters_and_marks_the_seeds():
    status = _status(
        {"n00009": "http://z:1", "n00001": "http://me:1", "n00005": "http://v:1", "n00002": "http://a:1"},
        voters=["n00001", "n00005"],
        seeds={"n00002": "http://a:1"},
    )
    result = await _peers(status).of(NODE)
    assert [p.id for p in result.peers] == ["n00001", "n00005", "n00002", "n00009"]
    assert result.peers[0].this_node and result.peers[0].voter and not result.peers[3].voter
    assert result.peers[2].seed and result.total == 4 and result.advertise_url == "http://adv:8420"


@pytest.mark.asyncio
async def test_a_long_list_is_paged_and_searchable():
    addresses = {f"n{i:05d}": f"http://10.1.{i // 250}.{i % 250}:8420" for i in range(1, 451)}
    status = _status(addresses, voters=["n00001"])
    first = await _peers(status).of(NODE, limit=200)
    assert (first.total, len(first.peers), first.offset) == (450, 200, 0)
    last = await _peers(status).of(NODE, offset=400, limit=200)
    assert len(last.peers) == 50 and last.peers[-1].id == "n00450"

    found = await _peers(status).of(NODE, query="10.1.1.")
    expected = [a for a in addresses.values() if "10.1.1." in a]
    assert found.total == len(expected) and all("10.1.1." in p.address for p in found.peers)
    assert (await _peers(status).of(NODE, query="N00007")).total == 1
    assert (await _peers(status).of(NODE, limit=10_000)).limit == MAX_PAGE


@pytest.mark.asyncio
async def test_only_a_node_of_this_cluster_is_asked_and_a_node_without_consensus_has_no_list():
    seen: list = []
    with pytest.raises(NodeNotAllowedError, match="not a node of this cluster"):
        await _peers(_status({}, []), seen).of("http://evil.example:8420")
    assert seen == []
    assert (await _peers(None).of(NODE)).total == 0


@pytest.mark.asyncio
async def test_the_endpoint_maps_a_stranger_to_404(db):
    with pytest.raises(HTTPException) as caught:
        await cluster_admin.get_peers("http://evil.example:8420", db=db, _admin=None)
    assert caught.value.status_code == 404


@pytest.mark.asyncio
async def test_every_address_is_listed_once_and_older_identities_are_counted_not_repeated():
    """Each reset gives a new id; an address must be listed once with older identities counted."""
    status = _status(
        {
            "n00001": "http://me:1",
            "n00002": "http://there:1",
            "old-a": "http://there:1",
            "old-b": "http://there:1/",
            "old-c": "http://gone:1",
        },
        voters=["n00001"],
    ) | {"nodes": {"n00001": "http://me:1", "n00002": "http://there:1"}}
    rows = (await _peers(status).of(NODE)).peers
    assert [(r.id, r.kind, r.old_ids) for r in rows] == [
        ("n00001", "voter", 0),
        ("n00002", "learner", 2),
        ("old-c", "old", 0),
    ]
    assert rows[0].this_node and len({r.address.rstrip("/") for r in rows}) == 3
