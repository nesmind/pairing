"""Settings > MCP servers: the tool servers (Model Context Protocol, streamable HTTP) the admin connects
so models can use their tools. Admin-only: this is where server credentials are entered."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.schemas import OkResponse
from app.schemas.mcp import McpEnabled, McpLimits, McpServerIn, McpServerOut, McpTestResult
from app.services import mcp_settings_service
from app.services.auth_service import require_admin
from app.services.mcp_server_service import McpServerError, McpServerService
from app.services.mcp_tool_catalog import McpToolCatalog

router = APIRouter(prefix="/api/settings/mcp-servers", tags=["mcp"])
enabled_router = APIRouter(prefix="/api/settings/mcp-enabled", tags=["mcp"])
limits_router = APIRouter(prefix="/api/settings/mcp-limits", tags=["mcp"])


def _status_for(error: McpServerError) -> int:
    return 404 if "not found" in str(error) else 409


@router.get("", response_model=list[McpServerOut])
async def list_servers(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    return [McpServerService.to_out(s) for s in await McpServerService(db).list()]


@router.post("/test", response_model=McpTestResult)
async def test_connection(
    body: McpServerIn,
    server_id: str | None = None,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Tries the form's values before saving. With `server_id` (editing) and no headers typed, the saved
    headers are used, since their values are never sent to the browser."""
    headers = body.headers
    if headers is None and server_id:
        try:
            headers = McpServerService.headers_of(await McpServerService(db).get(server_id))
        except McpServerError as exc:
            raise HTTPException(status_code=_status_for(exc), detail=str(exc)) from exc
    return await McpToolCatalog.probe(body.name, body.url, headers or {})


@router.post("", response_model=McpServerOut, status_code=201)
async def create_server(body: McpServerIn, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    try:
        return McpServerService.to_out(await McpServerService(db).create(body))
    except McpServerError as exc:
        raise HTTPException(status_code=_status_for(exc), detail=str(exc)) from exc


@router.put("/{server_id}", response_model=McpServerOut)
async def update_server(
    server_id: str, body: McpServerIn, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)
):
    try:
        return McpServerService.to_out(await McpServerService(db).update(server_id, body))
    except McpServerError as exc:
        raise HTTPException(status_code=_status_for(exc), detail=str(exc)) from exc


@router.delete("/{server_id}", response_model=OkResponse)
async def delete_server(server_id: str, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    try:
        await McpServerService(db).delete(server_id)
    except McpServerError as exc:
        raise HTTPException(status_code=_status_for(exc), detail=str(exc)) from exc
    return OkResponse()


@router.post("/{server_id}/test", response_model=McpTestResult)
async def test_server(server_id: str, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    try:
        server = await McpServerService(db).get(server_id)
    except McpServerError as exc:
        raise HTTPException(status_code=_status_for(exc), detail=str(exc)) from exc
    return await McpToolCatalog.probe(server.name, server.url, McpServerService.headers_of(server))


@enabled_router.get("", response_model=McpEnabled)
async def get_mcp_enabled(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    return McpEnabled(enabled=await mcp_settings_service.get_enabled(db))


@enabled_router.put("", response_model=McpEnabled)
async def set_mcp_enabled(body: McpEnabled, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    await mcp_settings_service.set_enabled(db, body.enabled)
    return body


@limits_router.get("", response_model=McpLimits)
async def get_limits(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    return await mcp_settings_service.get_limits(db)


@limits_router.put("", response_model=McpLimits)
async def set_limits(body: McpLimits, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    await mcp_settings_service.set_limits(db, body)
    return body
