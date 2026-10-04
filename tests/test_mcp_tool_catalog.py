import pytest

from app.schemas.mcp import McpServerIn
from app.services import mcp_tool_catalog
from app.services.mcp_server_service import McpServerService
from app.services.mcp_tool_catalog import McpToolCatalog
from tests.mcp_fake_server import FakeMcpServer


@pytest.fixture(scope="module")
def server():
    fake = FakeMcpServer().start()
    yield fake
    fake.stop()


@pytest.fixture(autouse=True)
def _fresh_cache():
    mcp_tool_catalog.clear_cache()


async def test_tools_are_prefixed_and_callable(db, server) -> None:
    await McpServerService(db).create(McpServerIn(name="fake", url=server.url))

    tools = {t.name: t for t in await McpToolCatalog(db).tools()}

    assert {"fake__add", "fake__echo"} <= set(tools)
    definition = tools["fake__add"].definition()
    assert definition["type"] == "function"
    assert definition["function"]["name"] == "fake__add"
    assert definition["function"]["parameters"]["properties"]["a"]["type"] == "integer"
    assert (await tools["fake__add"].call({"a": 1, "b": 2})).text == "3"


async def test_disabled_and_unreachable_servers_offer_nothing(db, server) -> None:
    service = McpServerService(db)
    await service.create(McpServerIn(name="off", url=server.url, enabled=False))
    await service.create(McpServerIn(name="dead", url="http://127.0.0.1:1/mcp"))

    assert await McpToolCatalog(db).tools() == []


async def test_two_servers_do_not_clash(db, server) -> None:
    service = McpServerService(db)
    await service.create(McpServerIn(name="one", url=server.url))
    await service.create(McpServerIn(name="two", url=server.url))

    names = [t.name for t in await McpToolCatalog(db).tools()]

    assert "one__add" in names and "two__add" in names
    assert len(names) == len(set(names))


async def test_listing_is_cached_until_the_server_is_edited(db, server, monkeypatch) -> None:
    created = await McpServerService(db).create(McpServerIn(name="fake", url=server.url))
    calls = 0
    real = mcp_tool_catalog.McpConnection.list_tools

    async def counting(self):
        nonlocal calls
        calls += 1
        return await real(self)

    monkeypatch.setattr(mcp_tool_catalog.McpConnection, "list_tools", counting)

    await McpToolCatalog(db).tools()
    await McpToolCatalog(db).tools()
    assert calls == 1

    await McpServerService(db).update(created.id, McpServerIn(name="fake", url=server.url, headers={"X": "1"}))
    await McpToolCatalog(db).tools()
    assert calls == 2


def test_long_and_odd_names_are_made_safe_and_unique() -> None:
    taken: set[str] = set()
    first = McpToolCatalog._unique_name("srv", "a" * 100, taken)
    taken.add(first)
    second = McpToolCatalog._unique_name("srv", "a" * 100, taken)

    assert len(first) == len(second) == 64
    assert first != second
    assert McpToolCatalog._unique_name("s", "get weather!", set()) == "s__get_weather_"


async def test_a_dead_server_is_not_retried_on_every_reply(db, monkeypatch) -> None:
    await McpServerService(db).create(McpServerIn(name="dead", url="http://127.0.0.1:1/mcp"))
    calls = 0
    real = mcp_tool_catalog.McpConnection.list_tools

    async def counting(self):
        nonlocal calls
        calls += 1
        return await real(self)

    monkeypatch.setattr(mcp_tool_catalog.McpConnection, "list_tools", counting)

    for _ in range(3):
        assert await McpToolCatalog(db).tools() == []
    assert calls == 1


def test_long_descriptions_are_clipped_in_the_definition_sent_to_the_model() -> None:
    schema = {"type": "object", "properties": {"q": {"type": "string", "description": "d" * 1000}}}
    tool = mcp_tool_catalog.ExposedTool("s__t", "s", "t", "x" * 2000, schema, None)

    function = tool.definition()["function"]

    assert len(function["description"]) <= 400
    assert len(function["parameters"]["properties"]["q"]["description"]) <= 160
    assert len(schema["properties"]["q"]["description"]) == 1000  # the original is untouched
