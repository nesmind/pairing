import json

import pytest

from app.services import inference_client, mcp_settings_service, tool_loop_service
from app.services.inference_client import InferenceError
from app.services.mcp_client import McpConnection, McpError
from app.services.mcp_tool_catalog import ExposedTool
from app.services.tool_call_sink import ToolCallSink
from app.services.tool_loop_service import ToolLoop
from tests.mcp_fake_server import FakeMcpServer


@pytest.fixture(scope="module")
def server():
    fake = FakeMcpServer().start()
    yield fake
    fake.stop()


def _tool(server, name: str = "add") -> ExposedTool:
    return ExposedTool(f"fake__{name}", "fake", name, "", {}, McpConnection(server.url))


class ScriptedEngine:
    """Stands in for chat_stream: each call plays the next scripted (text, tool_calls) turn."""

    def __init__(self, turns: list[tuple[str, list[dict]]]) -> None:
        self.turns = list(turns)
        self.seen: list[tuple[list[dict], dict]] = []

    async def __call__(self, model, messages, params):
        self.seen.append((json.loads(json.dumps(messages)), dict(params)))
        text, calls = self.turns.pop(0)
        sink = params.get("tool_sink")
        if sink is not None:
            sink.calls.extend(calls)
        yield text


async def _collect(loop: ToolLoop) -> list:
    return [item async for item in loop.run("m", [{"role": "user", "content": "hi"}], {"temperature": 0.1})]


def _call(name: str, args: dict, call_id: str | None = None) -> dict:
    call = {"function": {"name": name, "arguments": args}}
    if call_id:
        call["id"] = call_id
    return call


async def test_a_tool_call_runs_and_its_result_goes_back_to_the_model(server, monkeypatch) -> None:
    engine = ScriptedEngine([("Checking.", [_call("fake__add", {"a": 2, "b": 3})]), ("It is 5.", [])])
    monkeypatch.setattr(inference_client, "chat_stream", engine)

    items = await _collect(ToolLoop([_tool(server)]))

    text = "".join(i for i in items if isinstance(i, str))
    events = [i["tool"] for i in items if isinstance(i, dict)]
    assert text == "Checking.\n\nIt is 5."
    assert len(events) == 1
    assert (events[0]["server"], events[0]["tool"], events[0]["result"]) == ("fake", "add", "5")
    assert events[0]["is_error"] is False
    second_messages = engine.seen[1][0]
    assert second_messages[-2]["tool_calls"][0]["id"] == events[0]["id"]
    assert second_messages[-1] == {
        "role": "tool",
        "content": "5",
        "tool_name": "fake__add",
        "tool_call_id": events[0]["id"],
    }
    assert isinstance(engine.seen[0][1]["tool_sink"], ToolCallSink)


async def test_json_string_arguments_and_unknown_tools(server, monkeypatch) -> None:
    calls = [
        {"id": "x", "function": {"name": "fake__add", "arguments": '{"a": 1, "b": 1}'}},
        _call("nope__tool", {}),
    ]
    monkeypatch.setattr(inference_client, "chat_stream", ScriptedEngine([("", calls), ("done", [])]))

    events = [i["tool"] for i in await _collect(ToolLoop([_tool(server)])) if isinstance(i, dict)]

    assert events[0]["result"] == "2" and events[0]["id"] == "x"
    assert events[1]["is_error"] is True and "Unknown tool" in events[1]["result"]


async def test_an_unreachable_server_becomes_an_error_result(monkeypatch) -> None:
    dead = ExposedTool("d__t", "d", "t", "", {}, McpConnection("http://127.0.0.1:1/mcp", timeout=3))
    monkeypatch.setattr(inference_client, "chat_stream", ScriptedEngine([("", [_call("d__t", {})]), ("sorry", [])]))

    events = [i["tool"] for i in await _collect(ToolLoop([dead])) if isinstance(i, dict)]

    assert events[0]["is_error"] is True and events[0]["result"].startswith("Tool call failed")


async def test_rounds_are_capped_and_the_last_one_offers_no_tools(server, monkeypatch) -> None:
    endless = [("", [_call("fake__add", {"a": 1, "b": 1})]) for _ in range(2)] + [("final", [])]
    engine = ScriptedEngine(endless)
    monkeypatch.setattr(inference_client, "chat_stream", engine)

    items = await _collect(ToolLoop([_tool(server)], max_rounds=2))

    assert len([i for i in items if isinstance(i, dict)]) == 2
    assert [("tools" in params) for _, params in engine.seen] == [True, True, False]


async def test_a_model_without_tool_support_falls_back_to_a_plain_chat(server, monkeypatch) -> None:
    seen_params: list[dict] = []

    async def refuses_tools(model, messages, params):
        seen_params.append(params)
        if "tools" in params:
            raise InferenceError("registry.ollama.ai/library/x does not support tools")
        yield "plain answer"
        return

    monkeypatch.setattr(inference_client, "chat_stream", refuses_tools)

    items = await _collect(ToolLoop([_tool(server)]))

    assert items == ["plain answer"]
    assert "tools" not in seen_params[-1]


async def test_other_engine_errors_still_propagate(server, monkeypatch) -> None:
    async def broken(model, messages, params):
        raise InferenceError("engine down")
        yield ""

    monkeypatch.setattr(inference_client, "chat_stream", broken)

    with pytest.raises(InferenceError):
        await _collect(ToolLoop([_tool(server)]))


@pytest.mark.parametrize(
    ("use_tools", "have_tools", "capabilities", "expect_loop"),
    [
        (False, True, ["completion", "tools"], False),
        (True, False, ["completion", "tools"], False),
        (True, True, ["completion"], False),
        (True, True, ["completion", "tools"], True),
        (True, True, None, True),
    ],
)
async def test_tool_stream_or_none_gates(db, server, monkeypatch, use_tools, have_tools, capabilities, expect_loop):
    async def fake_tools(self):
        return [_tool(server)] if have_tools else []

    async def fake_models():
        entry = {"name": "m"}
        if capabilities is not None:
            entry["capabilities"] = capabilities
        return [entry]

    monkeypatch.setattr(tool_loop_service.McpToolCatalog, "tools", fake_tools)
    monkeypatch.setattr(inference_client, "list_models", fake_models)
    monkeypatch.setattr(tool_loop_service, "_capabilities", None)

    stream = await tool_loop_service.tool_stream_or_none(db, "m", [], {"use_tools": use_tools})

    assert (stream is not None) is expect_loop


async def test_global_switch_off_means_no_tool_loop(db, server, monkeypatch) -> None:
    async def fake_tools(self):
        return [_tool(server)]

    monkeypatch.setattr(tool_loop_service.McpToolCatalog, "tools", fake_tools)
    await mcp_settings_service.set_enabled(db, False)

    assert await tool_loop_service.tool_stream_or_none(db, "m", [], {"use_tools": True}) is None


def test_mcp_error_is_exported() -> None:
    assert issubclass(McpError, Exception)


async def test_capabilities_are_cached_between_replies(db, server, monkeypatch) -> None:
    calls = []

    async def fake_tools(self):
        return [_tool(server)]

    async def fake_models():
        calls.append(1)
        return [{"name": "m", "capabilities": ["tools"]}]

    monkeypatch.setattr(tool_loop_service.McpToolCatalog, "tools", fake_tools)
    monkeypatch.setattr(inference_client, "list_models", fake_models)
    monkeypatch.setattr(tool_loop_service, "_capabilities", None)

    for _ in range(3):
        assert await tool_loop_service.tool_stream_or_none(db, "m", [], {"use_tools": True}) is not None
    assert len(calls) == 1


async def test_a_long_tool_result_is_cut_to_the_admins_limit(server, monkeypatch) -> None:
    calls = [_call("fake__big", {})]
    monkeypatch.setattr(inference_client, "chat_stream", ScriptedEngine([("", calls), ("done", [])]))

    items = await _collect(ToolLoop([_tool(server, "big")], max_result_chars=500))

    result = next(i for i in items if isinstance(i, dict))["tool"]["result"]
    assert result.startswith("x" * 500) and "truncated" in result and len(result) < 600


async def test_an_identical_repeated_call_in_one_turn_runs_once(server, monkeypatch) -> None:
    calls = [
        _call("fake__add", {"a": 1, "b": 1}),
        _call("fake__add", {"b": 1, "a": 1}),
        _call("fake__add", {"a": 2, "b": 2}),
    ]
    monkeypatch.setattr(inference_client, "chat_stream", ScriptedEngine([("", calls), ("done", [])]))

    events = [i["tool"] for i in await _collect(ToolLoop([_tool(server)])) if isinstance(i, dict)]

    assert [e["result"] for e in events] == ["2", "4"]
