"""Admin-configured retention windows for the poller-written tables
app.services.retention_poller sweeps — split out into its own file for
the same file-size reason as app.services.chat_settings_service (see
that module's own docstring): app.services.settings_service is already
at CLAUDE.md's line cap.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import SYSTEM_OWNER_ID, AppSetting
from app.schemas import RetentionSettings

RETENTION_SETTINGS_KEY = "data_retention_days"

# Same defaults app.services.retention_poller shipped with before this became admin-configurable.
DEFAULT_TELEMETRY_RETENTION_DAYS = 30
DEFAULT_SYSTEM_METRICS_RETENTION_DAYS = 3


async def get_retention_settings(db: AsyncSession) -> RetentionSettings:
    """Read fresh on every call (not cached) — app.services.retention_poller.prune_once calls this at the start
    of every sweep, the same "an admin's change takes effect on the next run, not just after a restart"
    reasoning as chat_settings_service.get_reply_timeout_seconds. A stored value outside RetentionSettings' own
    [1, 365] bounds (only possible via a hand-edited row, since set_retention_settings below and the API layer's
    own schema validation both enforce this range) falls back to the defaults rather than applying a nonsense
    window."""
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, RETENTION_SETTINGS_KEY))
    if row:
        try:
            return RetentionSettings(**row.value)
        except (TypeError, ValueError):
            pass
    return RetentionSettings(
        telemetry_days=DEFAULT_TELEMETRY_RETENTION_DAYS,
        system_metrics_days=DEFAULT_SYSTEM_METRICS_RETENTION_DAYS,
    )


async def set_retention_settings(db: AsyncSession, settings: RetentionSettings) -> None:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, RETENTION_SETTINGS_KEY))
    value = settings.model_dump()
    if row is None:
        db.add(AppSetting(owner_id=SYSTEM_OWNER_ID, key=RETENTION_SETTINGS_KEY, value=value))
    else:
        row.value = value
    await db.commit()
