"""Tools only where the active engine can pass them to the model (EngineCapabilities.tools): a connector such as
Cloudflare hides the chat's Tools button and never starts the tool loop."""

from types import SimpleNamespace

import pytest

from app.routers import mcp_tools
from app.services import engine_service, tool_loop_service
from app.services.engines.registry import registry


@pytest.fixture(autouse=True)
def reset_engine():
    engine_service._cached_engine = engine_service.DEFAULT_ENGINE
    yield
    engine_service._cached_engine = engine_service.DEFAULT_ENGINE


def test_only_the_local_engines_pass_tools():
    supports = {engine.name: engine.capabilities.tools for engine in registry.all()}
    assert supports == {"ollama": True, "matricxon": True, "runpod": False, "cloudflare": False}


@pytest.mark.asyncio
async def test_chat_tools_are_off_on_a_connector_engine(db):
    conversation = SimpleNamespace(params={})
    engine_service._cached_engine = "matricxon"
    assert (await mcp_tools._chat_tools(db, conversation)).enabled is True
    engine_service._cached_engine = "cloudflare"
    assert (await mcp_tools._chat_tools(db, conversation)).enabled is False


@pytest.mark.asyncio
async def test_the_tool_loop_does_not_start_on_a_connector_engine(db):
    engine_service._cached_engine = "cloudflare"
    params = {"use_tools": True, "conversation_id": "c1"}
    assert await tool_loop_service.tool_stream_or_none(db, "@cf/m", [], params) is None
