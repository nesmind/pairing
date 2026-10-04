"""A tiny real MCP server (streamable HTTP) run in a thread, for tests."""

import socket
import threading
import time

import uvicorn
from mcp.server.fastmcp import FastMCP


def build_app() -> FastMCP:
    mcp = FastMCP("fake", host="127.0.0.1")

    @mcp.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @mcp.tool()
    def echo(text: str) -> str:
        """Echo text back."""
        return text

    @mcp.tool()
    def boom() -> str:
        """Always fails."""
        raise ValueError("kaboom")

    @mcp.tool()
    def big() -> str:
        """A very long result."""
        return "x" * 50_000

    return mcp


class FakeMcpServer:
    def __init__(self) -> None:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}/mcp"
        config = uvicorn.Config(build_app().streamable_http_app(), host="127.0.0.1", port=self.port, log_level="error")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def start(self) -> "FakeMcpServer":
        self._thread.start()
        for _ in range(100):
            if self._server.started:
                return self
            time.sleep(0.05)
        raise RuntimeError("fake MCP server did not start")

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5)
