from app.services.tool_call_sink import ToolCallSink, add_tools_to_payload


def test_sink_collects_calls_from_any_chunk_and_empties_on_take() -> None:
    sink = ToolCallSink()
    sink.add_from_chunk({"message": {"content": "hi"}})
    sink.add_from_chunk({"message": {"content": "", "tool_calls": [{"function": {"name": "a"}}]}})
    sink.add_from_chunk({"done": True})

    assert [c["function"]["name"] for c in sink.take()] == ["a"]
    assert sink.take() == []


def test_tools_only_go_in_the_payload_when_given() -> None:
    payload: dict = {}
    add_tools_to_payload(payload, {})
    assert "tools" not in payload
    add_tools_to_payload(payload, {"tools": [{"type": "function"}]})
    assert payload["tools"] == [{"type": "function"}]
