import pytest

from app.schemas.mcp import McpServerIn
from app.services.mcp_server_service import McpServerError, McpServerService


def _body(name: str = "files", **kw) -> McpServerIn:
    return McpServerIn(name=name, url="http://localhost:9000/mcp", **kw)


async def test_create_encrypts_headers_and_hides_values(db) -> None:
    service = McpServerService(db)

    server = await service.create(_body(headers={"Authorization": "Bearer s3cret"}))

    assert "s3cret" not in server.headers_encrypted
    assert service.headers_of(server) == {"Authorization": "Bearer s3cret"}
    out = service.to_out(server)
    assert out.header_names == ["Authorization"]
    assert "s3cret" not in out.model_dump_json()


async def test_update_keeps_headers_unless_sent(db) -> None:
    service = McpServerService(db)
    server = await service.create(_body(headers={"X-Key": "1"}))

    await service.update(server.id, _body(name="renamed"))
    assert service.headers_of(server) == {"X-Key": "1"}
    assert server.name == "renamed"

    await service.update(server.id, _body(name="renamed", headers={}))
    assert service.headers_of(server) == {}


async def test_duplicate_name_is_rejected(db) -> None:
    service = McpServerService(db)
    await service.create(_body())

    with pytest.raises(McpServerError):
        await service.create(_body())


async def test_list_can_filter_enabled_and_delete(db) -> None:
    service = McpServerService(db)
    on = await service.create(_body("a"))
    await service.create(_body("b", enabled=False))

    assert [s.name for s in await service.list()] == ["a", "b"]
    assert [s.name for s in await service.list(only_enabled=True)] == ["a"]
    await service.delete(on.id)
    assert [s.name for s in await service.list()] == ["b"]
    with pytest.raises(McpServerError):
        await service.get(on.id)


@pytest.mark.parametrize("name", ["", "has space", "x" * 41, "a/b"])
def test_bad_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError):
        McpServerIn(name=name, url="http://h/mcp")


@pytest.mark.parametrize("url", ["ftp://h/mcp", "http://user:pw@h/mcp", "localhost:9000", "http://"])
def test_bad_urls_are_rejected(url: str) -> None:
    with pytest.raises(ValueError):
        McpServerIn(name="ok", url=url)
