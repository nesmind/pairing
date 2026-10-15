"""Tests for services/default_model_settings.py, the one place every "which model is the default" read/write goes
through. list_models is monkeypatched on this module's own binding (not inference_client's origin; module-split
import-binding gotcha), so no real Ollama/Matricxon call.
"""

import pytest

from app.services import default_model_settings as svc
from app.services.default_model_settings import DefaultModelSettings


async def _async_return(value):
    return value


# for_new_users


@pytest.mark.asyncio
async def test_for_new_users_none_when_nothing_installed(db):
    assert await DefaultModelSettings(db).for_new_users([]) is None


@pytest.mark.asyncio
async def test_for_new_users_falls_back_to_the_first_installed_when_unconfigured(db):
    assert await DefaultModelSettings(db).for_new_users(["model-a", "model-b"]) == "model-a"


@pytest.mark.asyncio
async def test_for_new_users_returns_the_configured_choice_once_set(db):
    settings = DefaultModelSettings(db)
    await settings.set_for_new_users("model-b")
    assert await settings.for_new_users(["model-a", "model-b"]) == "model-b"


@pytest.mark.asyncio
async def test_for_new_users_falls_back_when_the_configured_one_was_uninstalled(db):
    settings = DefaultModelSettings(db)
    await settings.set_for_new_users("model-gone")
    assert await settings.for_new_users(["model-a", "model-b"]) == "model-a"


# vision


@pytest.mark.asyncio
async def test_vision_none_when_nothing_installed(db, monkeypatch):
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))
    assert await DefaultModelSettings(db).vision() is None


@pytest.mark.asyncio
async def test_vision_filters_to_the_vision_capability_only(db, monkeypatch):
    monkeypatch.setattr(
        svc,
        "list_models",
        lambda: _async_return(
            [
                {"name": "chat-only", "capabilities": ["completion"]},
                {"name": "vision-model", "capabilities": ["completion", "vision"]},
            ]
        ),
    )
    assert await DefaultModelSettings(db).vision() == "vision-model"


@pytest.mark.asyncio
async def test_vision_returns_the_configured_choice_once_set(db, monkeypatch):
    monkeypatch.setattr(
        svc,
        "list_models",
        lambda: _async_return(
            [
                {"name": "vision-a", "capabilities": ["vision"]},
                {"name": "vision-b", "capabilities": ["vision"]},
            ]
        ),
    )
    settings = DefaultModelSettings(db)
    await settings.set_vision("vision-b")
    assert await settings.vision() == "vision-b"


# embedding


@pytest.mark.asyncio
async def test_embedding_none_when_nothing_installed(db, monkeypatch):
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))
    assert await DefaultModelSettings(db).embedding() is None


@pytest.mark.asyncio
async def test_embedding_returns_the_configured_choice_once_set(db, monkeypatch):
    monkeypatch.setattr(
        svc,
        "list_models",
        lambda: _async_return(
            [
                {"name": "embed-a", "capabilities": ["embedding"]},
                {"name": "embed-b", "capabilities": ["embedding"]},
            ]
        ),
    )
    settings = DefaultModelSettings(db)
    await settings.set_embedding("embed-b")
    assert await settings.embedding() == "embed-b"


# isolation: setting one kind never affects the other two


@pytest.mark.asyncio
async def test_setting_one_default_never_affects_the_other_two(db, monkeypatch):
    """The point of consolidating the three: they're independent AppSetting rows, with no shared state."""
    monkeypatch.setattr(
        svc,
        "list_models",
        lambda: _async_return(
            [
                {"name": "shared-tag", "capabilities": ["completion", "vision", "embedding"]},
            ]
        ),
    )
    settings = DefaultModelSettings(db)
    await settings.set_for_new_users("shared-tag")

    assert await settings.for_new_users(["shared-tag"]) == "shared-tag"
    # vision/embedding were never configured: still "first installed with that capability", not inheriting new-users'
    # value.
    assert await settings.vision() == "shared-tag"
    assert await settings.embedding() == "shared-tag"

    await settings.set_vision("shared-tag")
    await settings.set_embedding("other-tag-not-installed")
    # Setting vision/embedding must not change what for_new_users returns.
    assert await settings.for_new_users(["shared-tag"]) == "shared-tag"


# chat_roles


@pytest.mark.asyncio
async def test_chat_roles_lists_every_role_a_model_fills(db):
    installed = [{"name": "chat-a", "capabilities": ["completion"]}, {"name": "see-b", "capabilities": ["vision"]}]
    roles = await DefaultModelSettings(db).chat_roles(installed)
    assert roles == {"chat-a": ["Default for new users"], "see-b": ["Vision"]}


@pytest.mark.asyncio
async def test_chat_roles_merges_both_roles_on_one_model(db):
    installed = [{"name": "both", "capabilities": ["completion", "vision"]}]
    assert await DefaultModelSettings(db).chat_roles(installed) == {"both": ["Default for new users", "Vision"]}


@pytest.mark.asyncio
async def test_chat_roles_follows_the_configured_vision_model(db):
    installed = [{"name": n, "capabilities": ["completion", "vision"]} for n in ("a", "b")]
    settings = DefaultModelSettings(db)
    await settings.set_vision("b")
    assert await settings.chat_roles(installed) == {"a": ["Default for new users"], "b": ["Vision"]}
