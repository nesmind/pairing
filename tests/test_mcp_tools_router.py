from app.routers import mcp_tools
from app.schemas.mcp import McpServerIn
from app.services import mcp_settings_service, mcp_tool_catalog
from app.services.mcp_server_service import McpServerService
from tests.mcp_fake_server import FakeMcpServer


async def test_lists_tools_of_enabled_servers_only(db) -> None:
    fake = FakeMcpServer().start()
    try:
        mcp_tool_catalog.clear_cache()
        service = McpServerService(db)
        await service.create(McpServerIn(name="on", url=fake.url))
        await service.create(McpServerIn(name="off", url=fake.url, enabled=False))

        tools = await mcp_tools.list_tools(db=db, _user=None)
    finally:
        fake.stop()

    names = [t.name for t in tools]
    assert "on__add" in names
    assert not any(n.startswith("off__") for n in names)
    assert next(t for t in tools if t.name == "on__add").description == "Add two numbers."


async def test_no_servers_means_no_tools(db) -> None:
    assert await mcp_tools.list_tools(db=db, _user=None) == []


async def test_global_switch_off_hides_every_tool(db) -> None:
    fake = FakeMcpServer().start()
    try:
        mcp_tool_catalog.clear_cache()
        await McpServerService(db).create(McpServerIn(name="on", url=fake.url))
        await mcp_settings_service.set_enabled(db, False)

        assert await mcp_tools.list_tools(db=db, _user=None) == []
        await mcp_settings_service.set_enabled(db, True)
        assert await mcp_tools.list_tools(db=db, _user=None) != []
    finally:
        fake.stop()
