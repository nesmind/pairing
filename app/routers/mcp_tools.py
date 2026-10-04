"""What MCP tools the chat page can offer: any signed-in user may ask (it shows or hides the Tools toggle)."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.schemas.mcp import McpToolOut
from app.services import mcp_settings_service
from app.services.auth_service import get_current_user
from app.services.mcp_tool_catalog import McpToolCatalog

router = APIRouter(prefix="/api/mcp", tags=["mcp"])


@router.get("/tools", response_model=list[McpToolOut])
async def list_tools(db: AsyncSession = Depends(get_db), _user: User = Depends(get_current_user)):
    if not await mcp_settings_service.get_enabled(db):
        return []
    return [McpToolOut(name=t.name, description=t.description) for t in await McpToolCatalog(db).tools()]
