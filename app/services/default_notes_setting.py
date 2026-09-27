"""
Whether a user wants their persona/rules/skill default notes automatically active on every
brand-new chat/channel they start (Settings > Account, under the Theme section). Split out of
app/services/note_service.py (already at CLAUDE.md's line cap) rather than added there, and kept
as its own tiny file rather than folded into settings_service.py (itself already at that same
cap, per its own history — see app/services/theme_service.py's identical split for the same
reason) — mirrors that file's own get_x/set_x per-user setting pair in shape.

Deliberately has no dependency on app.services.note_service (which depends on
app.services.channel_service) so both app.services.conversation_service and
app.services.channel_service can import this directly with no circular-import workaround needed
— see note_service.seed_initial_disabled_notes, the one place this value is actually turned into
a new conversation's own initial `disabled_default_notes`.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AppSetting, User

DEFAULT_NOTES_ENABLED_KEY = "default_notes_enabled"


async def get_default_notes_enabled(db: AsyncSession, user: User) -> bool:
    """Off for a new account with no saved row yet — a deliberate choice (2026-09-27): before
    this setting existed, every account got all 3 persona/rules/skill defaults on for every new
    chat with no way to opt out up front, only per-chat afterward via the chat page's own icons
    (see app.schemas.note.NoteSlotOut.active's own docstring — a disabled slot shows faded, not
    hidden, so it's still there to turn back on)."""
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
