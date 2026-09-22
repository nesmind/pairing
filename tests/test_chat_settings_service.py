"""Unit tests for app/services/chat_settings_service.py: channel delivery
mode and the reply timeout (see tests/test_title_service.py for title
mode's own two-mode behavior — this file only covers the plain getter/
setter round-trip for it, alongside the other settings that live in this
module)."""

import pytest

from app.models import SYSTEM_OWNER_ID, AppSetting
from app.services import chat_settings_service


@pytest.mark.asyncio
async def test_get_title_mode_defaults_to_simple(db):
    assert await chat_settings_service.get_title_mode(db) == "simple"


@pytest.mark.asyncio
async def test_set_title_mode_round_trips(db):
    await chat_settings_service.set_title_mode(db, "smart")
    assert await chat_settings_service.get_title_mode(db) == "smart"

    await chat_settings_service.set_title_mode(db, "simple")
    assert await chat_settings_service.get_title_mode(db) == "simple"


@pytest.mark.asyncio
async def test_get_channel_delivery_mode_defaults_to_cheap(db):
    assert await chat_settings_service.get_channel_delivery_mode(db) == "cheap"


@pytest.mark.asyncio
async def test_set_channel_delivery_mode_round_trips(db):
    await chat_settings_service.set_channel_delivery_mode(db, "real")
    assert await chat_settings_service.get_channel_delivery_mode(db) == "real"

    await chat_settings_service.set_channel_delivery_mode(db, "cheap")
    assert await chat_settings_service.get_channel_delivery_mode(db) == "cheap"


@pytest.mark.asyncio
async def test_get_channel_delivery_mode_falls_back_on_an_invalid_stored_value(db):
    """Defensive against a hand-edited or stale row holding a value this
    version of the app no longer recognizes — falls back to the default
    rather than surfacing a nonsense mode to the frontend."""
    db.add(
        AppSetting(
            owner_id=SYSTEM_OWNER_ID, key=chat_settings_service.CHANNEL_DELIVERY_MODE_KEY, value={"mode": "bogus"}
        )
    )
    await db.commit()

    assert await chat_settings_service.get_channel_delivery_mode(db) == "cheap"


@pytest.mark.asyncio
async def test_get_reply_timeout_seconds_defaults_to_300(db):
    assert await chat_settings_service.get_reply_timeout_seconds(db) == 300


@pytest.mark.asyncio
async def test_set_reply_timeout_seconds_round_trips(db):
    await chat_settings_service.set_reply_timeout_seconds(db, 60)
    assert await chat_settings_service.get_reply_timeout_seconds(db) == 60

    await chat_settings_service.set_reply_timeout_seconds(db, 300)
    assert await chat_settings_service.get_reply_timeout_seconds(db) == 300


@pytest.mark.asyncio
async def test_set_reply_timeout_seconds_zero_means_no_timeout(db):
    """0 is a real, intentional value (see
    chat_settings_service.DEFAULT_REPLY_TIMEOUT_SECONDS's own docstring
    for why), not just an edge case that happens to be allowed."""
    await chat_settings_service.set_reply_timeout_seconds(db, 0)
    assert await chat_settings_service.get_reply_timeout_seconds(db) == 0


@pytest.mark.asyncio
async def test_get_reply_timeout_seconds_falls_back_on_an_invalid_stored_value(db):
    db.add(
        AppSetting(
            owner_id=SYSTEM_OWNER_ID,
            key=chat_settings_service.REPLY_TIMEOUT_SECONDS_KEY,
            value={"timeout_seconds": -5},
        )
    )
    await db.commit()

    assert await chat_settings_service.get_reply_timeout_seconds(db) == 300


@pytest.mark.asyncio
async def test_get_vision_reply_timeout_seconds_defaults_to_300(db):
    assert await chat_settings_service.get_vision_reply_timeout_seconds(db) == 300


@pytest.mark.asyncio
async def test_set_vision_reply_timeout_seconds_round_trips(db):
    await chat_settings_service.set_vision_reply_timeout_seconds(db, 900)
    assert await chat_settings_service.get_vision_reply_timeout_seconds(db) == 900

    await chat_settings_service.set_vision_reply_timeout_seconds(db, 300)
    assert await chat_settings_service.get_vision_reply_timeout_seconds(db) == 300


@pytest.mark.asyncio
async def test_vision_reply_timeout_seconds_is_independent_of_the_text_one(db):
    """The whole point of splitting these into two settings — changing one must never affect the other."""
    await chat_settings_service.set_reply_timeout_seconds(db, 60)
    await chat_settings_service.set_vision_reply_timeout_seconds(db, 1200)

    assert await chat_settings_service.get_reply_timeout_seconds(db) == 60
    assert await chat_settings_service.get_vision_reply_timeout_seconds(db) == 1200


@pytest.mark.asyncio
async def test_set_vision_reply_timeout_seconds_zero_means_no_timeout(db):
    await chat_settings_service.set_vision_reply_timeout_seconds(db, 0)
    assert await chat_settings_service.get_vision_reply_timeout_seconds(db) == 0


@pytest.mark.asyncio
async def test_get_vision_reply_timeout_seconds_falls_back_on_an_invalid_stored_value(db):
    db.add(
        AppSetting(
            owner_id=SYSTEM_OWNER_ID,
            key=chat_settings_service.VISION_REPLY_TIMEOUT_SECONDS_KEY,
            value={"timeout_seconds": -5},
        )
    )
    await db.commit()

    assert await chat_settings_service.get_vision_reply_timeout_seconds(db) == 300
