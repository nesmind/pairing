"""Stores the MCP servers an admin connected (headers encrypted at rest)."""

import json

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import McpServer
from app.schemas.mcp import McpServerIn, McpServerOut
from app.services import secret_crypto


class McpServerError(Exception):
    """Unknown server, or a duplicate name."""


class McpServerService:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def list(self, only_enabled: bool = False) -> list[McpServer]:
        query = select(McpServer).order_by(McpServer.name)
        if only_enabled:
            query = query.where(McpServer.enabled.is_(True))
        return list((await self._db.execute(query)).scalars())

    async def get(self, server_id: str) -> McpServer:
        server = await self._db.get(McpServer, server_id)
        if server is None:
            raise McpServerError("MCP server not found")
        return server

    async def create(self, body: McpServerIn) -> McpServer:
        server = McpServer(
            name=body.name, url=body.url, enabled=body.enabled, call_timeout_seconds=body.call_timeout_seconds
        )
        server.headers_encrypted = self._encrypt(body.headers)
        self._db.add(server)
        await self._commit()
        return server

    async def update(self, server_id: str, body: McpServerIn) -> McpServer:
        server = await self.get(server_id)
        server.name, server.url, server.enabled = body.name, body.url, body.enabled
        server.call_timeout_seconds = body.call_timeout_seconds
        if body.headers is not None:
            server.headers_encrypted = self._encrypt(body.headers)
        await self._commit()
        return server

    async def delete(self, server_id: str) -> None:
        await self._db.delete(await self.get(server_id))
        await self._db.commit()

    @staticmethod
    def headers_of(server: McpServer) -> dict[str, str]:
        if not server.headers_encrypted:
            return {}
        plain = secret_crypto.decrypt(server.headers_encrypted)
        return json.loads(plain) if plain else {}

    @classmethod
    def to_out(cls, server: McpServer) -> McpServerOut:
        return McpServerOut(
            id=server.id,
            name=server.name,
            url=server.url,
            header_names=sorted(cls.headers_of(server)),
            enabled=server.enabled,
            call_timeout_seconds=server.call_timeout_seconds,
        )

    @staticmethod
    def _encrypt(headers: dict[str, str] | None) -> str | None:
        return secret_crypto.encrypt(json.dumps(headers)) if headers else None

    async def _commit(self) -> None:
        try:
            await self._db.commit()
        except IntegrityError as exc:
            await self._db.rollback()
            raise McpServerError("An MCP server with that name already exists") from exc
