"""Settings > System > Context window: the one `num_ctx` for every chat and channel. It was a per-chat
slider, which let chats of one model disagree and hurt the prompt cache. Read per reply, so changes apply
from the next message.
"""

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
