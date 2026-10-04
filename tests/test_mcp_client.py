"""McpConnection against a real in-process MCP server."""

import pytest

from app.services.mcp_client import MAX_RESULT_CHARS, McpConnection, McpError
from tests.mcp_fake_server import FakeMcpServer


@pytest.fixture(scope="module")
def server():
    fake = FakeMcpServer().start()
    yield fake
    fake.stop()


async def test_list_tools(server) -> None:
    tools = await McpConnection(server.url).list_tools()

    names = {t.name for t in tools}
    assert {"add", "echo", "boom", "big"} <= names
    add = next(t for t in tools if t.name == "add")
    assert add.description == "Add two numbers."
    assert add.input_schema["properties"]["a"]["type"] == "integer"


async def test_call_tool(server) -> None:
    result = await McpConnection(server.url).call_tool("add", {"a": 2, "b": 3})

    assert (result.text, result.is_error) == ("5", False)


async def test_tool_error_is_a_result_not_an_exception(server) -> None:
    result = await McpConnection(server.url).call_tool("boom", {})

    assert result.is_error is True
    assert "kaboom" in result.text


async def test_long_result_is_truncated(server) -> None:
    result = await McpConnection(server.url).call_tool("big", {})

    assert len(result.text) < MAX_RESULT_CHARS + 200
    assert "truncated" in result.text


async def test_unreachable_server_raises_mcp_error() -> None:
    with pytest.raises(McpError):
        await McpConnection("http://127.0.0.1:1/mcp", timeout=3).list_tools()


async def test_unknown_tool_is_an_error_result_or_mcp_error(server) -> None:
    try:
        result = await McpConnection(server.url).call_tool("nope", {})
    except McpError:
        return
    assert result.is_error is True
