"""Unit tests for app/services/default_notes_setting.py's per-user getter/setter — see
tests/test_note_service.py's seed_initial_disabled_notes tests for how this value actually turns
into a new conversation's own disabled_default_notes."""

import pytest

from app.services import default_notes_setting


@pytest.mark.asyncio
async def test_get_default_notes_enabled_defaults_to_false_when_unset(db, user):
    assert await default_notes_setting.get_default_notes_enabled(db, user) is False


@pytest.mark.asyncio
async def test_set_default_notes_enabled_round_trips(db, user):
    await default_notes_setting.set_default_notes_enabled(db, user, True)
    assert await default_notes_setting.get_default_notes_enabled(db, user) is True

    await default_notes_setting.set_default_notes_enabled(db, user, False)
    assert await default_notes_setting.get_default_notes_enabled(db, user) is False


@pytest.mark.asyncio
async def test_default_notes_enabled_is_isolated_per_user(db, user, admin_user):
    await default_notes_setting.set_default_notes_enabled(db, user, True)

    assert await default_notes_setting.get_default_notes_enabled(db, user) is True
    assert await default_notes_setting.get_default_notes_enabled(db, admin_user) is False
