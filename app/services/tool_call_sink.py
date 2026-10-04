"""Carries tool calls out of an engine's chat stream without changing its text-only contract.

`chat_stream` still yields text chunks only. A caller that offers tools puts `tools` (the tool
definitions) and a `ToolCallSink` in `params` (like `request_id`); the engine client adds each
`tool_calls` entry it sees to the sink, and the caller reads them once the stream ends."""

from typing import Any


class ToolCallSink:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def add_from_chunk(self, data: dict[str, Any]) -> None:
        """Takes one parsed NDJSON line of an Ollama/Matricxon chat stream."""
        for call in (data.get("message") or {}).get("tool_calls") or []:
            self.calls.append(call)

    def take(self) -> list[dict[str, Any]]:
        calls, self.calls = self.calls, []
        return calls


def add_tools_to_payload(payload: dict[str, Any], params: dict[str, Any]) -> None:
    if params.get("tools"):
        payload["tools"] = params["tools"]


def sink_from(params: dict[str, Any]) -> ToolCallSink | None:
    return params.get("tool_sink")
