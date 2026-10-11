"""Cluster id and joining an existing cluster."""

import httpx
import pytest

from app.schemas import MatricxonServerConfig
from app.schemas.cluster import ClusterHost
from app.services import matricxon_pool, server_pool_broadcast, settings_service
from app.services.cluster_join import ClusterJoin
from app.services.cluster_nodes import NodeNotAllowedError

NODE = "http://10.0.0.7:8420"
ID = "0123456789abcdef0123456789abcdef"


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    async def none(_path):
        return []

    monkeypatch.setattr(server_pool_broadcast, "broadcast_refresh", none)
    yield
    matricxon_pool.refresh_from_config(MatricxonServerConfig())


def _transport(status: dict | None):
    return httpx.MockTransport(lambda request: httpx.Response(200, json=status or {"enabled": False}))


def _status(cluster_id: str | None = ID) -> dict:
    paths = {"models_dir": "/mnt/m", "prompt_cache_dir": "/mnt/c"}
    return {"enabled": True, "cluster_id": cluster_id, "config": {"cluster/paths": paths}}


def test_the_id_is_the_one_the_newest_epoch_reports():
    old = ClusterHost(host="a", reachable=True, consensus={"epoch": 3, "cluster_id": "old"})
    new = ClusterHost(host="b", reachable=True, consensus={"epoch": 9, "cluster_id": ID})
    none = ClusterHost(host="c", reachable=True)
    assert ClusterJoin.id_of([old, new, none]) == ID and ClusterJoin.id_of([none]) is None


@pytest.mark.asyncio
async def test_a_pairing_joins_a_cluster_by_the_id_of_one_of_its_nodes(db):
    await ClusterJoin(db, _transport(_status())).join(NODE, ID.upper())
    config = await settings_service.get_matricxon_server_config(db)
    assert config.cluster_role == "seed" and config.mode == "remote" and config.remote_hosts == [NODE]
    assert config.models_path == "/mnt/m" and config.prompt_cache_path == "/mnt/c"


@pytest.mark.asyncio
async def test_another_clusters_id_or_a_node_without_one_is_refused_and_nothing_is_saved(db):
    for status, why in [
        (_status("f" * 32), "another cluster"),
        (_status(None), "no cluster id"),
        (None, "does not answer"),
    ]:
        with pytest.raises(NodeNotAllowedError, match=why):
            await ClusterJoin(db, _transport(status)).join(NODE, ID)
    assert (await settings_service.get_matricxon_server_config(db)).cluster_role == "off"


@pytest.mark.asyncio
async def test_a_pairing_that_already_has_cluster_mode_on_does_not_join_another(db):
    config = await settings_service.get_matricxon_server_config(db)
    config.cluster_role = "seed"
    await settings_service.set_matricxon_server_config(db, config)
    with pytest.raises(NodeNotAllowedError, match="already on"):
        await ClusterJoin(db, _transport(_status())).join(NODE, ID)


@pytest.mark.asyncio
async def test_leaving_turns_cluster_mode_off_here_and_keeps_every_node_as_a_plain_host(db):
    config = await settings_service.get_matricxon_server_config(db)
    config.mode, config.remote_hosts, config.cluster_hosts = "remote", [NODE], ["http://10.0.0.8:8420"]
    config.cluster_role, config.first_node, config.models_path = "seed", NODE, "/mnt/m"
    await settings_service.set_matricxon_server_config(db, config)
    await ClusterJoin(db).leave()
    left = await settings_service.get_matricxon_server_config(db)
    assert left.cluster_role == "off" and left.cluster_hosts == [] and left.first_node is None
    assert left.mode == "remote" and left.remote_hosts == [NODE, "http://10.0.0.8:8420"] and left.models_path is None


@pytest.mark.asyncio
async def test_leaving_a_cluster_with_a_local_matricxon_turns_it_into_a_remote_host(db):
    config = await settings_service.get_matricxon_server_config(db)
    config.mode, config.cluster_hosts, config.cluster_role = "local", [NODE], "seed"
    await settings_service.set_matricxon_server_config(db, config)
    await ClusterJoin(db).leave()
    left = await settings_service.get_matricxon_server_config(db)
    assert (
        left.mode == "remote" and matricxon_pool.LOCAL_MATRICXON_HOST in left.remote_hosts and NODE in left.remote_hosts
    )


@pytest.mark.asyncio
async def test_the_cluster_id_is_shown_only_while_cluster_mode_is_on_here(db, monkeypatch):
    from app.services.cluster_service import ClusterService

    async def hosts(_self, only=None):
        return [ClusterHost(host=NODE, reachable=True, consensus={"epoch": 1, "cluster_id": ID})]

    monkeypatch.setattr(ClusterService, "_read_hosts", hosts)
    monkeypatch.setattr(ClusterService, "_images_root", lambda _self: _images())
    config = await settings_service.get_matricxon_server_config(db)
    config.cluster_role = "seed"
    await settings_service.set_matricxon_server_config(db, config)
    assert (await ClusterService(db).state()).cluster_id == ID
    await ClusterJoin(db).leave()
    assert (await ClusterService(db).state()).cluster_id is None


async def _images():
    from pathlib import Path

    return Path("/nonexistent")


@pytest.mark.asyncio
async def test_a_protected_cluster_needs_its_admin_secret_to_join(db):
    protected = _status() | {"admin_auth": True}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "PUT":
            return httpx.Response(200 if "X-Matricxon-Signature" in request.headers else 401, json={})
        return httpx.Response(200, json=protected)

    transport = httpx.MockTransport(handler)
    with pytest.raises(NodeNotAllowedError, match="admin secret"):
        await ClusterJoin(db, transport).join(NODE, ID)
    assert (await settings_service.get_matricxon_server_config(db)).cluster_role == "off"

    def wrong(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401 if request.method == "PUT" else 200, json=protected)

    with pytest.raises(NodeNotAllowedError, match="not the cluster's admin secret"):
        await ClusterJoin(db, httpx.MockTransport(wrong)).join(NODE, ID, "nope")

    await ClusterJoin(db, transport).join(NODE, ID, "right")
    assert (await settings_service.get_matricxon_server_config(db)).admin_secret == "right"
    from app.services import matricxon_http

    matricxon_http.set_secret(None)
