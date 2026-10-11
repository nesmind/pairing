"""Whether a user's persona/rules/skill default notes are active on every new chat/channel (Settings >
Account, under Theme). Its own tiny file because note_service.py and settings_service.py are at the line
cap (like theme_service.py); same get_x/set_x per-user shape.

Doesn't depend on note_service (which depends on channel_service), so conversation_service and
channel_service import it with no circular workaround; note_service.seed_initial_disabled_notes turns it
into a new conversation's initial `disabled_default_notes`.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AppSetting, User

DEFAULT_NOTES_ENABLED_KEY = "default_notes_enabled"


async def get_default_notes_enabled(db: AsyncSession, user: User) -> bool:
    """Off for a new account with no saved row (deliberate). Before this setting every account got
    all 3 defaults on in every new chat, with per-chat opt-out only (NoteSlotOut.active: a disabled slot shows
    faded, not hidden).
    """
    row = await db.get(AppSetting, (user.id, DEFAULT_NOTES_ENABLED_KEY))
    return bool(row.value["enabled"]) if row else False


async def set_default_notes_enabled(db: AsyncSession, user: User, enabled: bool) -> None:
    row = await db.get(AppSetting, (user.id, DEFAULT_NOTES_ENABLED_KEY))
    value = {"enabled": enabled}
    if row is None:
        db.add(AppSetting(owner_id=user.id, key=DEFAULT_NOTES_ENABLED_KEY, value=value))
    else:
        row.value = value
    await db.commit()
