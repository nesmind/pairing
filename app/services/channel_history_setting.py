"""Settings > System > Channel replies: whether messages members post in a channel without asking the AI are
part of what the model reads on a later reply. On (the original behaviour) lets the model follow the whole
discussion; off keeps its prompt to the questions it was actually asked, which is shorter and cheaper."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import SYSTEM_OWNER_ID, AppSetting

KEY = "channel_plain_messages_to_model"


async def get_include_plain_messages(db: AsyncSession) -> bool:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, KEY))
    return True if row is None else bool(row.value.get("include", True))


async def set_include_plain_messages(db: AsyncSession, include: bool) -> None:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, KEY))
    if row is None:
        db.add(AppSetting(owner_id=SYSTEM_OWNER_ID, key=KEY, value={"include": include}))
    else:
        row.value = {"include": include}
    await db.commit()
