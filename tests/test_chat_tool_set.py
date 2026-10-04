"""A chat keeps the tools it started with; removed tools stay but fail clearly (app/services/chat_tool_set.py)."""

import pytest

from app.models import Conversation
from app.routers import mcp_tools
from app.services import inference_client, mcp_settings_service
from app.services.chat_tool_set import ChatTool, ChatToolSet
from app.services.mcp_client import McpConnection
from app.services.mcp_tool_catalog import ExposedTool, McpToolCatalog
from app.services.tool_loop_service import ToolLoop
from tests.test_tool_loop_service import ScriptedEngine, _call, _collect


def _live(*names: str) -> list[ExposedTool]:
    return [ExposedTool(f"s__{n}", "s", n, f"{n} tool", {}, McpConnection("http://127.0.0.1:1/mcp")) for n in names]


def test_snapshot_is_sorted_and_round_trips_to_chat_tools():
    snapshot = ChatToolSet.snapshot_of(_live("b", "a"))

    assert [s["name"] for s in snapshot] == ["s__a", "s__b"]
    tools = ChatToolSet.tools(snapshot, _live("a", "b"))
    assert [t.available for t in tools] == [True, True]


def test_a_removed_tool_stays_in_the_list_and_a_new_one_is_only_suggested():
    snapshot = ChatToolSet.snapshot_of(_live("a", "b"))
    live_now = _live("a", "c")

    tools = ChatToolSet.tools(snapshot, live_now)
    new = ChatToolSet.new_tools(snapshot, live_now)

    assert [(t.name, t.available) for t in tools] == [("s__a", True), ("s__b", False)]
    assert [t.name for t in new] == ["s__c"]


@pytest.mark.asyncio
async def test_calling_a_removed_tool_tells_the_model_it_is_gone():
    gone = ChatTool("s__b", "s", "b", {"type": "function", "function": {"name": "s__b"}}, None)

    result = await gone.call({})

    assert result.is_error and "no longer available" in result.text and "s__b" in result.text


@pytest.mark.asyncio
async def test_the_first_use_saves_the_list_and_later_changes_do_not_alter_it(db, user):
    conversation = Conversation(owner_id=user.id, params={})
    db.add(conversation)
    await db.commit()
    tool_set = ChatToolSet(db)

    first = await tool_set.for_chat(conversation.id, [], _live("a", "b"))
    await db.refresh(conversation)
    saved = conversation.params["tool_snapshot"]
    later = await tool_set.for_chat(conversation.id, saved, _live("a"))  # b removed, c would be new

    assert [t.name for t in first] == ["s__a", "s__b"]
    assert [(t.name, t.available) for t in later] == [("s__a", True), ("s__b", False)]
    assert await tool_set.for_chat(conversation.id, [], []) == []  # nothing offered, nothing saved


@pytest.mark.asyncio
async def test_a_loop_over_a_removed_tool_returns_the_error_to_the_model(monkeypatch):
    gone = ChatTool("s__b", "s", "b", {"type": "function", "function": {"name": "s__b", "parameters": {}}}, None)
    engine = ScriptedEngine([("", [_call("s__b", {})]), ("done", [])])
    monkeypatch.setattr(inference_client, "chat_stream", engine)

    items = await _collect(ToolLoop([gone]))

    event = next(i for i in items if isinstance(i, dict))["tool"]
    assert event["is_error"] and "no longer available" in event["result"]
    assert "no longer available" in engine.seen[1][0][-1]["content"]  # the model reads it in round two


@pytest.mark.asyncio
async def test_the_chat_tools_endpoint_lists_unavailable_and_new_tools(db, user, monkeypatch):
    async def fake_tools(self):
        return _live("a", "c")

    monkeypatch.setattr(McpToolCatalog, "tools", fake_tools)
    saved = ChatToolSet.snapshot_of(_live("a", "b"))
    conversation = Conversation(owner_id=user.id, params={"tool_snapshot": saved})
    db.add(conversation)
    await db.commit()
    await mcp_settings_service.set_enabled(db, True)

    out = await mcp_tools.chat_tools(conversation.id, db=db, user=user)

    assert out.frozen is True
    assert [(t.name, t.available) for t in out.tools] == [("s__a", True), ("s__b", False)]
    assert [t.name for t in out.new_tools] == ["s__c"]

    refreshed = await mcp_tools.refresh_chat_tools(conversation.id, db=db, user=user)
    assert [t.name for t in refreshed.tools] == ["s__a", "s__c"] and refreshed.new_tools == []


@pytest.mark.asyncio
async def test_a_params_update_never_changes_the_saved_tools(db, user):
    from app.schemas import ConversationUpdate, GenerationParams
    from app.services import conversation_service

    saved = ChatToolSet.snapshot_of(_live("a"))
    conversation = Conversation(owner_id=user.id, params={"tool_snapshot": saved})
    db.add(conversation)
    await db.commit()
    forged = GenerationParams(tool_snapshot=[{"name": "evil", "server": "x", "tool": "y", "spec": {}}], use_tools=True)

    await conversation_service.update_conversation(db, conversation, ConversationUpdate(params=forged), user)

    assert conversation.params["tool_snapshot"] == saved and conversation.params["use_tools"] is True
