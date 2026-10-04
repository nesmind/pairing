"""The admin's global MCP on/off switch and tool-round cap (Settings > System). Off means no chat offers
tools to a model, whatever each chat's Tools switch says. Read fresh from the database on every call,
so it applies at once on every instance."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import SYSTEM_OWNER_ID, AppSetting
from app.schemas.mcp import McpLimits

MCP_ENABLED_KEY = "mcp_enabled"
LIMITS_KEY = "mcp_limits"
DEFAULT_MAX_TOOL_ROUNDS = 5
DEFAULT_MAX_RESULT_CHARS = 4000  # ~1000 tokens: a longer result makes every later round slow on CPU


async def get_enabled(db: AsyncSession) -> bool:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, MCP_ENABLED_KEY))
    return True if row is None else bool(row.value.get("enabled", True))


async def set_enabled(db: AsyncSession, enabled: bool) -> None:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, MCP_ENABLED_KEY))
    if row is None:
        db.add(AppSetting(owner_id=SYSTEM_OWNER_ID, key=MCP_ENABLED_KEY, value={"enabled": enabled}))
    else:
        row.value = {"enabled": enabled}
    await db.commit()


async def get_limits(db: AsyncSession) -> McpLimits:
    """Tool rounds per reply and characters of a tool result the model reads (every MCP server alike)."""
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, LIMITS_KEY))
    value = row.value if row else {}
    rounds, chars = value.get("rounds"), value.get("result_chars")
    return McpLimits(
        rounds=rounds if isinstance(rounds, int) and 1 <= rounds <= 20 else DEFAULT_MAX_TOOL_ROUNDS,
        result_chars=chars if isinstance(chars, int) and 500 <= chars <= 20000 else DEFAULT_MAX_RESULT_CHARS,
    )


async def set_limits(db: AsyncSession, limits: McpLimits) -> None:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, LIMITS_KEY))
    if row is None:
        db.add(AppSetting(owner_id=SYSTEM_OWNER_ID, key=LIMITS_KEY, value=limits.model_dump()))
    else:
        row.value = limits.model_dump()
    await db.commit()
