import httpx
import pytest

from app.schemas import MatricxonServerConfig
from app.services import matricxon_http, matricxon_pool
from app.services.cluster_errors import NodeNotAllowedError
from app.services.cluster_secret import ClusterSecret


@pytest.fixture(autouse=True)
def _reset_secret():
    yield
    matricxon_http.set_secret(None)


@pytest.mark.asyncio
async def test_requests_are_signed_only_while_a_secret_is_set():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    async with matricxon_http.client(transport=transport) as client:
        await client.post("http://n:1/api/x?a=1", json={"k": 1})
        matricxon_http.set_secret("s3")
        await client.post("http://n:1/api/x?a=1", json={"k": 1})
    assert "X-Matricxon-Signature" not in seen[0].headers
    signed = seen[1]
    expected = matricxon_http.signature(
        "s3",
        "POST",
        "/api/x?a=1",
        signed.content,
        signed.headers["X-Matricxon-Time"],
        signed.headers["X-Matricxon-Nonce"],
    )
    assert signed.headers["X-Matricxon-Signature"] == expected


def test_pool_hands_chats_only_to_compute_nodes():
    config = MatricxonServerConfig(
        mode="remote",
        remote_hosts=["http://a:1"],
        cluster_hosts=["http://b:1", "http://pod:1"],
        voter_hosts=["http://pod:1"],
    )
    matricxon_pool.refresh_from_config(config)
    assert matricxon_pool.get_effective_hosts() == ["http://a:1", "http://b:1", "http://pod:1"]
    assert {matricxon_pool.pick_host() for _ in range(6)} <= {"http://a:1", "http://b:1"}


@pytest.mark.asyncio
async def test_changing_the_secret_needs_every_node_and_rolls_back(db):
    matricxon_pool.refresh_from_config(MatricxonServerConfig(mode="remote", remote_hosts=["http://a:1", "http://b:1"]))
    puts: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={})
        host = request.url.host
        puts.append((host, request.content.decode()))
        return httpx.Response(500 if host == "b" and '"new"' in request.content.decode() else 200, json={})

    with pytest.raises(NodeNotAllowedError, match="b:1"):
        await ClusterSecret(db, httpx.MockTransport(handler)).set("new")
    assert ("a", '{"secret":"new"}') in puts and ("a", '{"secret":""}') in puts  # a was put back
    # nothing changed for pAIring
    from app.services import settings_service

    assert (await settings_service.get_matricxon_server_config(db)).admin_secret is None


@pytest.mark.asyncio
async def test_a_silent_node_blocks_the_change(db):
    matricxon_pool.refresh_from_config(MatricxonServerConfig(mode="remote", remote_hosts=["http://a:1"]))

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    with pytest.raises(NodeNotAllowedError, match="must answer"):
        await ClusterSecret(db, httpx.MockTransport(handler)).set("x")


@pytest.mark.asyncio
async def test_the_secret_is_stored_encrypted_and_survives_a_round_trip(db):
    from sqlalchemy import select

    from app.models import AppSetting
    from app.services import settings_service

    config = MatricxonServerConfig(admin_secret="topsecret")
    await settings_service.set_matricxon_server_config(db, config)
    row = (
        await db.execute(select(AppSetting).where(AppSetting.key == settings_service.MATRICXON_SERVER_CONFIG_KEY))
    ).scalar_one()
    assert "topsecret" not in str(row.value)
    assert (await settings_service.get_matricxon_server_config(db)).admin_secret == "topsecret"
