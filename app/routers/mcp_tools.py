"""What MCP tools the chat page can offer: any signed-in user may ask (it shows or hides the Tools toggle)."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.schemas.mcp import McpChatTool, McpChatTools, McpToolOut
from app.services import channel_service, conversation_service, mcp_settings_service
from app.services.auth_service import get_current_user
from app.services.chat_tool_set import ChatToolSet
from app.services.mcp_tool_catalog import McpToolCatalog

router = APIRouter(prefix="/api/mcp", tags=["mcp"])


@router.get("/tools", response_model=list[McpToolOut])
async def list_tools(db: AsyncSession = Depends(get_db), _user: User = Depends(get_current_user)):
    if not await mcp_settings_service.get_enabled(db):
        return []
    return [McpToolOut(name=t.name, description=t.description) for t in await McpToolCatalog(db).tools()]


async def _chat_tools(db: AsyncSession, conversation) -> McpChatTools:
    enabled = await mcp_settings_service.get_enabled(db)
    live = await McpToolCatalog(db).tools() if enabled else []
    saved = (conversation.params or {}).get("tool_snapshot") or []
    if not saved:
        return McpChatTools(
            enabled=enabled, frozen=False, tools=[McpChatTool(name=t.name, description=t.description) for t in live]
        )
    return McpChatTools(
        enabled=enabled,
        frozen=True,
        tools=[
            McpChatTool(name=t.name, description=t.spec["function"].get("description", ""), available=t.available)
            for t in ChatToolSet.tools(saved, live)
        ],
        new_tools=[McpToolOut(name=t.name, description=t.description) for t in ChatToolSet.new_tools(saved, live)],
    )


@router.get("/conversations/{conversation_id}/tools", response_model=McpChatTools)
async def chat_tools(conversation_id: str, db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    """The tools this chat offers, those an admin removed since (still listed, marked unavailable), and new ones."""
    conversation = await conversation_service.get_accessible_conversation_or_404(db, conversation_id, user)
    return await _chat_tools(db, conversation)


@router.post("/conversations/{conversation_id}/tools/refresh", response_model=McpChatTools)
async def refresh_chat_tools(
    conversation_id: str, db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)
):
    """Replaces the chat's saved tools with the current ones - its next reply re-reads the whole chat once."""
    conversation = await conversation_service.get_accessible_conversation_or_404(db, conversation_id, user)
    if conversation.channel_id is not None and not channel_service.can_manage_channel_conversation(conversation, user):
        raise HTTPException(status_code=403, detail="Only admins and this channel's managers can change its tools.")
    live = await McpToolCatalog(db).tools() if await mcp_settings_service.get_enabled(db) else []
    snapshot = ChatToolSet.snapshot_of(live)
    await ChatToolSet(db).save(conversation_id, snapshot)
    return await _chat_tools(db, conversation)
