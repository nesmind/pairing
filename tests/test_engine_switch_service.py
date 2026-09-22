"""Unit tests for app/services/engine_switch_service.py — clear_stale_default_models, called by
app/routers/engine_admin.py's PUT handler only when the active engine actually changed (see
tests/test_engine_admin_router.py for that integration point). list_models is monkeypatched on this module's own
imported binding, not app.services.inference_client directly (see the "module-split import-binding gotcha" this
codebase already tracks), so no real Ollama/Matricxon call happens here."""

import pytest

from app.models import SYSTEM_OWNER_ID, AppSetting
from app.services import default_model_settings, engine_switch_service, model_catalog_service, settings_service
from app.services.default_model_settings import DefaultModelSettings


async def _async_return(value):
    return value


def _installed(*tags):
    return [{"name": tag, "capabilities": ["completion"]} for tag in tags]


@pytest.mark.asyncio
async def test_clears_a_system_wide_default_that_is_no_longer_installed(db, monkeypatch):
    await settings_service.set_default_model(db, SYSTEM_OWNER_ID, "stale-tag")
    monkeypatch.setattr(engine_switch_service, "list_models", lambda: _async_return(_installed("new-tag")))

    await engine_switch_service.clear_stale_default_models(db)

    assert await db.get(AppSetting, (SYSTEM_OWNER_ID, settings_service.DEFAULT_MODEL_KEY)) is None


@pytest.mark.asyncio
async def test_clears_a_users_own_default_that_is_no_longer_installed(db, user, monkeypatch):
    await settings_service.set_default_model(db, user.id, "stale-tag")
    monkeypatch.setattr(engine_switch_service, "list_models", lambda: _async_return(_installed("new-tag")))

    await engine_switch_service.clear_stale_default_models(db)

    assert await db.get(AppSetting, (user.id, settings_service.DEFAULT_MODEL_KEY)) is None


@pytest.mark.asyncio
async def test_clears_a_stale_default_for_new_users_row(db, monkeypatch):
    await model_catalog_service.set_default_model_for_new_users(db, "stale-tag")
    monkeypatch.setattr(engine_switch_service, "list_models", lambda: _async_return(_installed("new-tag")))

    await engine_switch_service.clear_stale_default_models(db)

    assert await db.get(AppSetting, (SYSTEM_OWNER_ID, DefaultModelSettings.FOR_NEW_USERS_KEY)) is None


@pytest.mark.asyncio
async def test_leaves_a_default_untouched_when_it_is_still_installed(db, monkeypatch):
    """The real case this whole function exists to avoid breaking: a tag that happens to already be installed
    under both engines (e.g. shared dev/test models — see model_catalog_service's own module docstring) must
    survive an engine switch unchanged, not get wiped unconditionally."""
    await settings_service.set_default_model(db, SYSTEM_OWNER_ID, "shared-tag")
    monkeypatch.setattr(engine_switch_service, "list_models", lambda: _async_return(_installed("shared-tag")))

    await engine_switch_service.clear_stale_default_models(db)

    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, settings_service.DEFAULT_MODEL_KEY))
    assert row is not None
    assert row.value["model"] == "shared-tag"


@pytest.mark.asyncio
async def test_ignores_installed_models_with_no_completion_capability(db, monkeypatch):
    """An embedding-only model reaching /api/tags must not count as "still installed" for a chat default — same
    "completion" capability filter installed_chat_models() itself already applies."""
    await settings_service.set_default_model(db, SYSTEM_OWNER_ID, "embed-only-tag")
    monkeypatch.setattr(
        engine_switch_service,
        "list_models",
        lambda: _async_return([{"name": "embed-only-tag", "capabilities": ["embedding"]}]),
    )

    await engine_switch_service.clear_stale_default_models(db)

    assert await db.get(AppSetting, (SYSTEM_OWNER_ID, settings_service.DEFAULT_MODEL_KEY)) is None


@pytest.mark.asyncio
async def test_leaves_unrelated_settings_rows_untouched(db, monkeypatch):
    await settings_service.set_default_model(db, SYSTEM_OWNER_ID, "stale-tag")
    await model_catalog_service.set_default_vision_model(db, "some-vision-tag")
    monkeypatch.setattr(engine_switch_service, "list_models", lambda: _async_return(_installed("new-tag")))

    await engine_switch_service.clear_stale_default_models(db)

    # default_vision_model has its own installed-check in get_default_vision_model already (see that function's
    # own docstring) — not one of the two keys this function touches.
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, default_model_settings._VISION_KEY))
    assert row is not None
    assert row.value["model"] == "some-vision-tag"
