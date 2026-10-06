"""Settings > System > Context window: the one `num_ctx` every chat and channel uses. It was a per-chat Behavior slider,
which let two chats of the same model disagree and made the prompt cache's job harder. Read at the start of each
reply, so a change applies from the next message with no restart."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import DEFAULT_GENERATION_PARAMS
from app.models import SYSTEM_OWNER_ID, AppSetting

KEY = "context_window_tokens"


async def get_num_ctx(db: AsyncSession) -> int:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, KEY))
    return int(row.value["num_ctx"]) if row else int(DEFAULT_GENERATION_PARAMS["num_ctx"])


async def set_num_ctx(db: AsyncSession, num_ctx: int) -> None:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, KEY))
    if row is None:
        db.add(AppSetting(owner_id=SYSTEM_OWNER_ID, key=KEY, value={"num_ctx": num_ctx}))
    else:
        row.value = {"num_ctx": num_ctx}
    await db.commit()
