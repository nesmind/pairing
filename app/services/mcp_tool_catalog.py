"""The tools of every enabled MCP server, shaped the way the model is offered them.

Tool names are `<server>__<tool>` so two servers can't clash and a call is routed back to the right
server. Each server's tool list is cached for a minute per instance (a cache miss only costs a request),
keyed on the server's saved settings so editing a server never serves a stale list. A server that fails is remembered
as empty for a few seconds, so a dead one doesn't add its timeout to every reply."""

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import McpServer
from app.schemas.mcp import McpTestResult, McpToolOut
from app.services.mcp_client import McpConnection, McpError, McpToolInfo, McpToolResult
from app.services.mcp_server_service import McpServerService

logger = logging.getLogger("llama_chat")

_CACHE_SECONDS = 60.0
_FAILURE_CACHE_SECONDS = 15.0
_MAX_NAME = 64
_MAX_TOOL_DESCRIPTION = 400  # tool definitions are re-sent (and re-read) every round, so keep them short
_MAX_PARAM_DESCRIPTION = 160
_cache: dict[str, tuple[float, tuple, list[McpToolInfo]]] = {}


@dataclass(frozen=True)
class ExposedTool:
    name: str  # what the model sees: <server>__<tool>
    server_name: str
    tool_name: str
    description: str
    schema: dict[str, Any]
    connection: McpConnection

    def definition(self) -> dict[str, Any]:
        """Ollama/OpenAI-style tool definition."""
        parameters = _shortened(self.schema) if self.schema else {"type": "object", "properties": {}}
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": _clip(self.description, _MAX_TOOL_DESCRIPTION),
                "parameters": parameters,
            },
        }

    async def call(self, arguments: dict[str, Any]) -> McpToolResult:
        return await self.connection.call_tool(self.tool_name, arguments)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _shortened(schema: Any) -> Any:
    """The schema with every long parameter `description` clipped (a copy; the cached one is untouched)."""
    if isinstance(schema, dict):
        return {
            key: _clip(value, _MAX_PARAM_DESCRIPTION)
            if key == "description" and isinstance(value, str)
            else _shortened(value)
            for key, value in schema.items()
        }
    if isinstance(schema, list):
        return [_shortened(item) for item in schema]
    return schema


class McpToolCatalog:
    def __init__(self, db: AsyncSession) -> None:
        self._service = McpServerService(db)

    async def tools(self) -> list[ExposedTool]:
        servers = await self._service.list(only_enabled=True)
        listings = await asyncio.gather(*(self._tools_of(s) for s in servers))
        exposed: list[ExposedTool] = []
        taken: set[str] = set()
        for server, listing in zip(servers, listings, strict=True):
            connection = McpConnection(server.url, self._service.headers_of(server))
            for info in listing:
                name = self._unique_name(server.name, info.name, taken)
                taken.add(name)
                exposed.append(
                    ExposedTool(name, server.name, info.name, info.description, info.input_schema, connection)
                )
        return exposed

    @staticmethod
    async def _tools_of(server: McpServer) -> list[McpToolInfo]:
        fingerprint = (server.url, server.headers_encrypted)
        cached = _cache.get(server.id)
        if cached and cached[0] > time.monotonic() and cached[1] == fingerprint:
            return cached[2]
        try:
            headers = McpServerService.headers_of(server)
            listing = await McpConnection(server.url, headers).list_tools()
        except McpError as exc:
            logger.warning("MCP server %r is unavailable: %s", server.name, exc)
            _cache[server.id] = (time.monotonic() + _FAILURE_CACHE_SECONDS, fingerprint, [])
            return []
        _cache[server.id] = (time.monotonic() + _CACHE_SECONDS, fingerprint, listing)
        return listing

    @staticmethod
    async def probe(name: str, url: str, headers: dict[str, str]) -> McpTestResult:
        """Connects right now (no cache) for the admin's "Test connection" button."""
        try:
            listing = await McpConnection(url, headers).list_tools()
        except McpError as exc:
            return McpTestResult(ok=False, error=str(exc))
        return McpTestResult(
            ok=True, tools=[McpToolOut(name=f"{name}__{t.name}", description=t.description) for t in listing]
        )

    @staticmethod
    def _unique_name(server: str, tool: str, taken: set[str]) -> str:
        base = re.sub(r"[^A-Za-z0-9_-]", "_", f"{server}__{tool}")[:_MAX_NAME]
        name, n = base, 1
        while name in taken:
            n += 1
            suffix = f"_{n}"
            name = base[: _MAX_NAME - len(suffix)] + suffix
        return name


def clear_cache() -> None:
    _cache.clear()
