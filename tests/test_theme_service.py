"""Unit tests for app/services/theme_service.py's per-user UI theme getter/setter."""

import pytest

from app.services import theme_service
from app.theme_config import DEFAULT_UI_THEME


@pytest.mark.asyncio
async def test_get_ui_theme_defaults_when_unset(db, user):
    assert await theme_service.get_ui_theme(db, user) == DEFAULT_UI_THEME


@pytest.mark.asyncio
async def test_set_ui_theme_round_trips(db, user):
    await theme_service.set_ui_theme(db, user, "sunshine")
    assert await theme_service.get_ui_theme(db, user) == "sunshine"

    await theme_service.set_ui_theme(db, user, "nord")
    assert await theme_service.get_ui_theme(db, user) == "nord"


@pytest.mark.asyncio
async def test_get_ui_theme_falls_back_to_default_for_an_invalid_saved_value(db, user):
    await theme_service.set_ui_theme(db, user, "a-theme-that-was-later-removed")
    assert await theme_service.get_ui_theme(db, user) == DEFAULT_UI_THEME


@pytest.mark.asyncio
async def test_ui_theme_is_isolated_per_user(db, user, admin_user):
    await theme_service.set_ui_theme(db, user, "bubblegum")

    assert await theme_service.get_ui_theme(db, user) == "bubblegum"
    assert await theme_service.get_ui_theme(db, admin_user) == DEFAULT_UI_THEME
