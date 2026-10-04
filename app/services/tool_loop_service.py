"""Runs a reply with tools: model -> tool calls -> run them on their MCP servers -> model again.

`ToolLoop.run` is a drop-in for `chat_stream` that also yields one dict per finished tool call, so the
caller can show and store it. After the admin's max rounds (Settings > System) the model is asked to answer without
tools, so a model that keeps calling tools can't run forever."""

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.services import inference_client, mcp_settings_service
from app.services.inference_client import InferenceError
from app.services.mcp_client import McpError
from app.services.mcp_tool_catalog import ExposedTool, McpToolCatalog
from app.services.tool_call_sink import ToolCallSink
from app.services.tool_result_clip import ToolResultClip


class ToolLoop:
    def __init__(
        self,
        tools: list[ExposedTool],
        max_rounds: int = mcp_settings_service.DEFAULT_MAX_TOOL_ROUNDS,
        max_result_chars: int = mcp_settings_service.DEFAULT_MAX_RESULT_CHARS,
    ) -> None:
        self._tools = {t.name: t for t in tools}
        self._definitions = [t.definition() for t in tools]
        self._max_rounds = max_rounds
        self._clip = ToolResultClip(max_result_chars)

    async def run(
        self, model: str, messages: list[dict], params: dict[str, Any]
    ) -> AsyncIterator[str | dict[str, Any]]:
        messages = list(messages)
        offer_tools = True
        for round_number in range(self._max_rounds + 1):
            offer_tools = offer_tools and round_number < self._max_rounds
            sink = ToolCallSink()
            call_params = {**params, "tools": self._definitions, "tool_sink": sink} if offer_tools else params
            text = ""
            try:
                async for chunk in inference_client.chat_stream(model, messages, call_params):
                    if round_number > 0 and not text:
                        yield "\n\n"
                    text += chunk
                    yield chunk
            except InferenceError as exc:
                if not (round_number == 0 and not text and "support tools" in str(exc)):
                    raise
                offer_tools = False  # this model can't use tools: answer plainly instead of failing
                async for chunk in inference_client.chat_stream(model, messages, params):
                    yield chunk
                return
            calls = sink.take() if offer_tools else []
            if not calls:
                return
            calls = _without_repeats(calls)
            for index, call in enumerate(calls):
                call["id"] = call.get("id") or f"call_{round_number}_{index}"
            messages.append({"role": "assistant", "content": text, "tool_calls": calls})
            for call in calls:
                event = await self._execute(call)
                yield {"tool": event}
                messages.append(
                    {
                        "role": "tool",
                        "content": event["result"],
                        "tool_name": event["name"],
                        "tool_call_id": event["id"],
                    }
                )

    async def _execute(self, call: dict[str, Any]) -> dict[str, Any]:
        function = call.get("function") or {}
        name = function.get("name", "")
        arguments = _arguments(function.get("arguments"))
        tool = self._tools.get(name)
        event: dict[str, Any] = {
            "id": call["id"],
            "name": name,
            "server": tool.server_name if tool else "",
            "tool": tool.tool_name if tool else name,
            "arguments": arguments,
        }
        if tool is None:
            return {**event, "result": f"Unknown tool: {name}", "is_error": True}
        try:
            result = await tool.call(arguments)
        except McpError as exc:
            return {**event, "result": f"Tool call failed: {exc}", "is_error": True}
        return {**event, "result": self._clip.fit(result.text), "is_error": result.is_error}


def _without_repeats(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A model sometimes repeats the very same call within one turn; running it twice only costs time."""
    seen: set[tuple[str, str]] = set()
    unique = []
    for call in calls:
        function = call.get("function") or {}
        key = (function.get("name", ""), json.dumps(_arguments(function.get("arguments")), sort_keys=True))
        if key not in seen:
            seen.add(key)
            unique.append(call)
    return unique


def _arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw) if isinstance(raw, str) and raw else {}
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def tool_stream_or_none(
    db: AsyncSession, model: str, messages: list[dict], params: dict[str, Any]
) -> AsyncIterator[str | dict[str, Any]] | None:
    """The tool-using reply stream for this request, or None when it should be a plain chat:
    the conversation didn't ask for tools, no MCP server offers any, or the model can't use them."""
    if not params.get("use_tools") or not await mcp_settings_service.get_enabled(db):
        return None
    tools = await McpToolCatalog(db).tools()
    if not tools or not await _model_can_use_tools(model):
        return None
    limits = await mcp_settings_service.get_limits(db)
    return ToolLoop(tools, limits.rounds, limits.result_chars).run(model, messages, params)


_CAPABILITY_SECONDS = 60.0
_capabilities: tuple[float, list[dict]] | None = None


async def _model_can_use_tools(model: str) -> bool:
    """False only when the engine reports capabilities for this model and "tools" isn't among them
    (Matricxon does; Ollama's model list doesn't, so there it is tried and falls back if refused).
    The model list is cached briefly so a tool reply doesn't pay for an extra engine call."""
    global _capabilities
    now = time.monotonic()
    if _capabilities is None or _capabilities[0] < now:
        try:
            _capabilities = (now + _CAPABILITY_SECONDS, await inference_client.list_models())
        except InferenceError:
            return True
    entry = next((m for m in _capabilities[1] if m.get("name") == model), None)
    capabilities = entry.get("capabilities") if entry else None
    return capabilities is None or "tools" in capabilities
