"""Talks to one MCP server over streamable HTTP.

Each call opens its own short session (initialize, one request, close) instead of keeping a
connection: nothing is shared between users or between pAIring instances, so any instance can serve
any reply, and a server restart never leaves a dead connection behind."""

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

MAX_RESULT_CHARS = 20_000  # hard cap; the admin's smaller limit is applied by the tool loop
_MAX_PARALLEL_CALLS = 8  # across all users on this instance

_slots = asyncio.Semaphore(_MAX_PARALLEL_CALLS)


class McpError(Exception):
    """The server was unreachable, timed out, or answered with a protocol error."""


@dataclass(frozen=True)
class McpToolInfo:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class McpServerInfo:
    """What a server says about itself when a session starts (the MCP `initialize` answer)."""

    name: str = ""
    version: str = ""
    protocol: str = ""
    instructions: str = ""
    capabilities: tuple[str, ...] = ()


@dataclass(frozen=True)
class McpInspection:
    server: McpServerInfo
    tools: list[McpToolInfo]
    seconds: float  # connect + initialize + list tools


@dataclass(frozen=True)
class McpToolResult:
    text: str
    is_error: bool


class McpConnection:
    def __init__(self, url: str, headers: dict[str, str] | None = None, timeout: float = 30.0) -> None:
        self._url = url
        self._headers = headers or {}
        self._timeout = timeout

    async def list_tools(self) -> list[McpToolInfo]:
        async def run(session: ClientSession) -> list[McpToolInfo]:
            listing = await session.list_tools()
            return [McpToolInfo(t.name, t.description or "", dict(t.inputSchema or {})) for t in listing.tools]

        return await self._with_session(run)

    async def inspect(self) -> McpInspection:
        """One short session: the server's own description plus its tools, and how long that took."""
        started = time.monotonic()

        async def run(session: ClientSession) -> list[McpToolInfo]:
            listing = await session.list_tools()
            return [McpToolInfo(t.name, t.description or "", dict(t.inputSchema or {})) for t in listing.tools]

        info: list[McpServerInfo] = []
        tools = await self._with_session(run, info)
        return McpInspection(info[0] if info else McpServerInfo(), tools, time.monotonic() - started)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpToolResult:
        async def run(session: ClientSession) -> McpToolResult:
            result = await session.call_tool(name, arguments)
            return McpToolResult(_result_text(result), bool(result.isError))

        return await self._with_session(run)

    async def _with_session(self, action: Any, info: list[McpServerInfo] | None = None) -> Any:
        try:
            async with _slots, asyncio.timeout(self._timeout):
                async with self._session(info) as session:
                    return await action(session)
        except TimeoutError as exc:
            raise McpError(f"timed out after {self._timeout:g}s") from exc
        except McpError:
            raise
        except Exception as exc:  # incl. the ExceptionGroup anyio raises from a failed connect
            raise McpError(_describe(exc)) from exc

    @asynccontextmanager
    async def _session(self, info: list[McpServerInfo] | None = None) -> AsyncIterator[ClientSession]:
        async with httpx.AsyncClient(headers=self._headers, timeout=self._timeout) as http:
            async with streamable_http_client(self._url, http_client=http) as (read, write, _):
                async with ClientSession(read, write) as session:
                    started = await session.initialize()
                    if info is not None:
                        info.append(_server_info(started))
                    yield session


def _server_info(started: Any) -> McpServerInfo:
    caps = getattr(started, "capabilities", None)
    names = tuple(
        n for n in ("tools", "resources", "prompts", "logging") if caps is not None and getattr(caps, n, None)
    )
    server = getattr(started, "serverInfo", None)
    return McpServerInfo(
        name=getattr(server, "name", "") or "",
        version=getattr(server, "version", "") or "",
        protocol=str(getattr(started, "protocolVersion", "") or ""),
        instructions=getattr(started, "instructions", "") or "",
        capabilities=names,
    )


def _result_text(result: Any) -> str:
    parts = [c.text if c.type == "text" else f"[{c.type} content not shown]" for c in result.content]
    text = "\n".join(parts) or (str(result.structuredContent) if result.structuredContent else "")
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + f"\n[truncated: result was longer than {MAX_RESULT_CHARS} characters]"
    return text


def _describe(exc: BaseException) -> str:
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return str(exc) or type(exc).__name__
