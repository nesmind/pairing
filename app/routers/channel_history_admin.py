"""Settings > System > Channel replies: do plain channel messages (posted without asking the AI) reach the model?"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.schemas.settings import ChannelPlainMessages
from app.services import channel_history_setting
from app.services.auth_service import require_admin

router = APIRouter(prefix="/api/settings/channel-plain-messages", tags=["settings"])


@router.get("", response_model=ChannelPlainMessages)
async def get_setting(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    return ChannelPlainMessages(include=await channel_history_setting.get_include_plain_messages(db))


@router.put("", response_model=ChannelPlainMessages)
async def set_setting(
    body: ChannelPlainMessages, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)
):
    await channel_history_setting.set_include_plain_messages(db, body.include)
    return body
