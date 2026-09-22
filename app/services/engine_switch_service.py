"""
What needs to happen elsewhere in the app right after an admin switches the active engine (see
app.services.engine_service, app/routers/engine_admin.py's PUT handler — the only caller) — split into its own
tiny module rather than folded into model_catalog_service.py/settings_service.py (both already at or past
CLAUDE.md's file-size cap) or engine_service.py itself (importing app.services.inference_client's list_models
from there would be circular: inference_client already imports engine_service to decide which client to
dispatch to).
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AppSetting
from app.services import settings_service
from app.services.default_model_settings import DefaultModelSettings
from app.services.inference_client import list_models

_DEFAULT_MODEL_KEYS = (settings_service.DEFAULT_MODEL_KEY, DefaultModelSettings.FOR_NEW_USERS_KEY)


async def clear_stale_default_models(db: AsyncSession) -> None:
    """Ollama and Matricxon each have their own separate model catalog (see
    model_catalog_service.resolve_installed_model's own docstring) — a default-model tag chosen under the
    previous engine routinely doesn't exist under the newly active one at all. Every *reader* of these two
    settings (settings_service.get_default_model, model_catalog_service.get_default_model_for_new_users) already
    has its own installed-check fallback chain, so this was never a hard functional break — but it did mean the
    Model tab kept showing (and every new account kept getting assigned) a specific, named model that had simply
    stopped existing, with nothing telling the admin that had happened.

    Deletes any default_model row — system-wide and every individual user's — plus the default-for-new-users
    row, whose stored tag isn't actually installed under whichever engine is active right now, leaving that spot
    genuinely unset (see app.static.js.settings's initModelTab, which shows an explicit "pick a default" banner
    once nothing here resolves to a real entry) rather than a phantom leftover choice. A tag that happens to
    already be installed under both engines (e.g. shared dev/test models) is left untouched — this only clears
    what's actually gone, never everything unconditionally."""
    installed = {m["name"] for m in await list_models() if "completion" in m.get("capabilities", [])}
    rows = (await db.execute(select(AppSetting).where(AppSetting.key.in_(_DEFAULT_MODEL_KEYS)))).scalars().all()
    changed = False
    for row in rows:
        if row.value.get("model") not in installed:
            await db.delete(row)
            changed = True
    if changed:
        await db.commit()
