"""Admin retention windows for the tables retention_poller sweeps. Own file for the same file-size reason as
chat_settings_service (settings_service is at the line cap).
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import SYSTEM_OWNER_ID, AppSetting
from app.schemas import RetentionSettings

RETENTION_SETTINGS_KEY = "data_retention_days"

# The defaults retention_poller shipped with before this was configurable.
DEFAULT_TELEMETRY_RETENTION_DAYS = 30
DEFAULT_SYSTEM_METRICS_RETENTION_DAYS = 3


async def get_retention_settings(db: AsyncSession) -> RetentionSettings:
    """Read fresh on every call: retention_poller.prune_once calls it each sweep, so changes apply at the next run
    (as chat_settings_service.get_reply_timeout_seconds). A stored value outside RetentionSettings' [1, 365]
    (only via a hand-edited row; set_retention_settings and the API schema enforce it) falls back to the defaults.
    """
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
