"""Unit tests for app/services/retention_settings_service.py."""

import pytest
from sqlalchemy import select

from app.models import SYSTEM_OWNER_ID, AppSetting
from app.schemas import RetentionSettings
from app.services import retention_settings_service


@pytest.mark.asyncio
async def test_get_retention_settings_returns_defaults_when_unset(db):
    settings = await retention_settings_service.get_retention_settings(db)

    assert settings.telemetry_days == retention_settings_service.DEFAULT_TELEMETRY_RETENTION_DAYS
    assert settings.system_metrics_days == retention_settings_service.DEFAULT_SYSTEM_METRICS_RETENTION_DAYS


@pytest.mark.asyncio
async def test_set_then_get_round_trips(db):
    await retention_settings_service.set_retention_settings(
        db, RetentionSettings(telemetry_days=10, system_metrics_days=2)
    )

    settings = await retention_settings_service.get_retention_settings(db)

    assert settings.telemetry_days == 10
    assert settings.system_metrics_days == 2


@pytest.mark.asyncio
async def test_set_retention_settings_is_idempotent_not_a_duplicate_row(db):
    await retention_settings_service.set_retention_settings(
        db, RetentionSettings(telemetry_days=10, system_metrics_days=2)
    )
    await retention_settings_service.set_retention_settings(
        db, RetentionSettings(telemetry_days=20, system_metrics_days=5)
    )

    settings = await retention_settings_service.get_retention_settings(db)
    assert settings.telemetry_days == 20
    assert settings.system_metrics_days == 5

    rows = (
        (
            await db.execute(
                select(AppSetting).where(AppSetting.key == retention_settings_service.RETENTION_SETTINGS_KEY)
            )
        )
        .scalars()
        .all()
    )
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_get_retention_settings_falls_back_to_defaults_for_a_malformed_stored_row(db):
    """Only reachable via a hand-edited row — set_retention_settings and the API's own schema validation both
    enforce RetentionSettings' shape, but a stored value can still end up malformed some other way (a manual DB
    edit, a future schema change) and must not crash the sweep."""
    db.add(AppSetting(owner_id=SYSTEM_OWNER_ID, key=retention_settings_service.RETENTION_SETTINGS_KEY, value={}))
    await db.commit()

    settings = await retention_settings_service.get_retention_settings(db)

    assert settings.telemetry_days == retention_settings_service.DEFAULT_TELEMETRY_RETENTION_DAYS
    assert settings.system_metrics_days == retention_settings_service.DEFAULT_SYSTEM_METRICS_RETENTION_DAYS
