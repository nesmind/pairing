"""The tools one chat offers its model: saved the first time the chat uses tools, then kept.

Tool definitions start the prompt, so changing them makes the engine re-read the whole chat. A chat keeps
its starting list so admin changes don't do that to every chat. A removed tool stays listed and a call
gets a "no longer available" result. New tools reach a chat only on request (`refresh`); new chats start
from the current tools.
"""

from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Conversation
from app.services.mcp_client import McpToolResult
from app.services.mcp_tool_catalog import ExposedTool

UNAVAILABLE = "The tool {name} is no longer available (an administrator removed it). Answer without it."


@dataclass(frozen=True)
class ChatTool:
    """A tool as this chat knows it: the saved definition plus the live tool that runs it (None once removed)."""

    name: str
    server_name: str
    tool_name: str
    spec: dict[str, Any]
    live: ExposedTool | None

    def definition(self) -> dict[str, Any]:
        return self.spec

    @property
    def available(self) -> bool:
        return self.live is not None

    async def call(self, arguments: dict[str, Any]) -> McpToolResult:
        if self.live is None:
            return McpToolResult(UNAVAILABLE.format(name=self.name), True)
        return await self.live.call(arguments)


class ChatToolSet:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    @staticmethod
    def snapshot_of(live: list[ExposedTool]) -> list[dict[str, Any]]:
        """The saved form of a tool list, in fixed order."""
        return [
            {"name": t.name, "server": t.server_name, "tool": t.tool_name, "spec": t.definition()}
            for t in sorted(live, key=lambda t: t.name)
        ]

    @staticmethod
    def tools(snapshot: list[dict[str, Any]], live: list[ExposedTool]) -> list[ChatTool]:
        live_by_name = {t.name: t for t in live}
        return [ChatTool(s["name"], s["server"], s["tool"], s["spec"], live_by_name.get(s["name"])) for s in snapshot]

    @staticmethod
    def new_tools(snapshot: list[dict[str, Any]], live: list[ExposedTool]) -> list[ExposedTool]:
        known = {s["name"] for s in snapshot}
        return [t for t in live if t.name not in known]

    async def for_chat(
        self, conversation_id: str | None, saved: list[dict[str, Any]], live: list[ExposedTool]
    ) -> list[ChatTool]:
        """This chat's tools. With nothing saved, saves the current ones (unless none)."""
        if not saved:
            if not live:
                return []
            saved = self.snapshot_of(live)
            if conversation_id:
                await self.save(conversation_id, saved)
        return self.tools(saved, live)

    async def save(self, conversation_id: str, snapshot: list[dict[str, Any]]) -> None:
        conversation = await self._db.get(Conversation, conversation_id)
        if conversation is not None:
            conversation.params = {**(conversation.params or {}), "tool_snapshot": snapshot}
            await self._db.commit()
