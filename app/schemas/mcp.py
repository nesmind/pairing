"""Schemas for the MCP servers admin API (Settings > MCP servers)."""

import re

from pydantic import BaseModel, Field, field_validator

_NAME = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


class McpServerIn(BaseModel):
    name: str
    url: str
    # Request headers sent to the server, e.g. {"Authorization": "Bearer ..."}. On update, omit to keep
    # the stored ones; send {} to clear them.
    headers: dict[str, str] | None = None
    enabled: bool = True
    call_timeout_seconds: int = Field(default=20, ge=1, le=300)  # per tool call on this server

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not _NAME.match(value):
            raise ValueError("name must be 1-40 letters, digits, '_' or '-'")
        return value

    @field_validator("url")
    @classmethod
    def _valid_url(cls, value: str) -> str:
        value = value.strip()
        if not re.match(r"^https?://[^/\s@]+(/\S*)?$", value):
            raise ValueError("url must be http(s)://host[:port]/path, without credentials")
        return value


class McpServerOut(BaseModel):
    id: str
    name: str
    url: str
    header_names: list[str] = Field(default_factory=list)  # values are never sent back
    enabled: bool
    call_timeout_seconds: int = 20


class McpEnabled(BaseModel):
    enabled: bool


class McpLimits(BaseModel):
    rounds: int = Field(ge=1, le=20)  # tool rounds per reply
    result_chars: int = Field(ge=500, le=20000)  # characters of one tool result the model reads


class McpToolOut(BaseModel):
    name: str  # as the model sees it: <server>__<tool>
    description: str = ""


class McpChatTool(McpToolOut):
    available: bool = True  # False once an admin removed it: still in the chat's list, but calls fail


class McpChatTools(BaseModel):
    """What one chat offers its model (see app.services.chat_tool_set) and what an admin added since."""

    enabled: bool = True  # the admin's global MCP switch
    frozen: bool  # the chat has its own saved list (it has used tools)
    tools: list[McpChatTool]
    new_tools: list[McpToolOut] = Field(default_factory=list)


class McpTestTool(McpToolOut):
    params: list[str] = Field(default_factory=list)  # parameter names, a trailing * marks a required one
    definition_chars: int = 0  # size of the definition the model reads (clipped the way chats send it)


class McpServerDetails(BaseModel):
    """What the server says about itself when the connection starts."""

    name: str = ""
    version: str = ""
    protocol: str = ""
    instructions: str = ""
    capabilities: list[str] = Field(default_factory=list)


class McpTestResult(BaseModel):
    ok: bool
    error: str | None = None
    tools: list[McpTestTool] = Field(default_factory=list)
    server: McpServerDetails | None = None
    seconds: float = 0.0  # connect, initialize and list the tools
    definition_tokens: int = 0  # rough size of all tool definitions, which the model re-reads every round
