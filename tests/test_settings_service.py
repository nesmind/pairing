"""Unit tests for app/services/settings_service.py's proxy-mode and
external-server-config getters/setters (see app/services/instance_pool.py
for where the cached copy of proxy-mode actually gets used for routing
decisions). See tests/test_chat_settings_service.py instead for title
mode, channel delivery mode, and the reply timeout — those moved to
app/services/chat_settings_service.py to keep this file under
CLAUDE.md's file-size rule."""

import pytest

from app.models import SYSTEM_OWNER_ID, AppSetting
from app.schemas import ComfyUIProcessConfig, OllamaServerConfig
from app.services import engine_service, settings_service


@pytest.mark.asyncio
async def test_get_proxy_mode_defaults_to_local(db):
    assert await settings_service.get_proxy_mode(db) == "local"


@pytest.mark.asyncio
async def test_set_proxy_mode_round_trips(db):
    await settings_service.set_proxy_mode(db, "proxy")
    assert await settings_service.get_proxy_mode(db) == "proxy"

    await settings_service.set_proxy_mode(db, "local")
    assert await settings_service.get_proxy_mode(db) == "local"


@pytest.mark.asyncio
async def test_get_ollama_server_config_defaults_to_local_with_nothing_set(db):
    config = await settings_service.get_ollama_server_config(db)

    assert config.mode == "local"
    assert config.remote_hosts == []
    assert config.num_parallel is None


@pytest.mark.asyncio
async def test_set_ollama_server_config_round_trips(db):
    await settings_service.set_ollama_server_config(
        db, OllamaServerConfig(mode="remote", remote_hosts=["http://10.0.0.1:11434", "http://10.0.0.2:11434"])
    )

    config = await settings_service.get_ollama_server_config(db)

    assert config.mode == "remote"
    assert config.remote_hosts == ["http://10.0.0.1:11434", "http://10.0.0.2:11434"]


def test_ollama_server_config_falls_back_to_local_when_remote_has_no_hosts():
    # Regression: local mode shows no Save button until something's
    # installed, so saving "remote" with an empty host list used to leave
    # an admin stuck with no UI path back to local — see the matching
    # validator on OllamaServerConfig itself.
    config = OllamaServerConfig(mode="remote", remote_hosts=[])
    assert config.mode == "local"


@pytest.mark.asyncio
async def test_set_ollama_server_config_normalizes_remote_with_no_hosts_to_local(db):
    await settings_service.set_ollama_server_config(db, OllamaServerConfig(mode="remote", remote_hosts=[]))

    config = await settings_service.get_ollama_server_config(db)

    assert config.mode == "local"


@pytest.mark.asyncio
async def test_get_ollama_server_config_self_heals_an_already_stuck_remote_row(db):
    # A row saved before this fix existed, stuck on "remote" with no
    # hosts — must self-heal back to "local" the moment it's read, not
    # just on the next save.
    db.add(
        AppSetting(
            owner_id=SYSTEM_OWNER_ID,
            key=settings_service.OLLAMA_SERVER_CONFIG_KEY,
            value={"mode": "remote", "remote_hosts": []},
        )
    )
    await db.commit()

    config = await settings_service.get_ollama_server_config(db)

    assert config.mode == "local"


@pytest.mark.asyncio
async def test_get_comfyui_config_defaults_mode_and_remote_hosts_for_a_pre_existing_row(db):
    """A row saved before mode/remote_hosts existed (just the three
    original path fields) must still load cleanly, with the new fields
    defaulting in — not a crash or a silently dropped old value."""
    db.add(
        AppSetting(
            owner_id=SYSTEM_OWNER_ID,
            key=settings_service.COMFYUI_CONFIG_KEY,
            value={"python_path": "/opt/venv/bin/python", "main_py_path": "/opt/ComfyUI/main.py", "extra_args": None},
        )
    )
    await db.commit()

    config = await settings_service.get_comfyui_config(db)

    assert config.python_path == "/opt/venv/bin/python"
    assert config.mode == "local"
    assert config.remote_hosts == []


@pytest.mark.asyncio
async def test_set_comfyui_config_round_trips_remote_mode(db):
    await settings_service.set_comfyui_config(
        db, ComfyUIProcessConfig(mode="remote", remote_hosts=["http://10.0.0.1:8188"])
    )

    config = await settings_service.get_comfyui_config(db)

    assert config.mode == "remote"
    assert config.remote_hosts == ["http://10.0.0.1:8188"]


def test_comfyui_config_falls_back_to_local_when_remote_has_no_hosts():
    # See test_ollama_server_config_falls_back_to_local_when_remote_has_no_hosts
    # — same regression, same fix, mirrored on ComfyUIProcessConfig.
    config = ComfyUIProcessConfig(mode="remote", remote_hosts=[])
    assert config.mode == "local"


@pytest.mark.asyncio
async def test_set_comfyui_config_normalizes_remote_with_no_hosts_to_local(db):
    await settings_service.set_comfyui_config(db, ComfyUIProcessConfig(mode="remote", remote_hosts=[]))

    config = await settings_service.get_comfyui_config(db)

    assert config.mode == "local"


# ---- default model (per-engine) --------------------------------------------
# Real bug found live, 2026-09-22: a single shared "default_model" key meant switching the active engine had
# to explicitly wipe it (see the now-removed app.services.engine_switch_service), losing the admin's actual
# preference outright rather than just no longer being able to display it. default_model_key scopes the
# stored value per engine instead, so neither switching nor a restart ever loses it.


@pytest.fixture(autouse=True)
def _reset_engine_cache():
    engine_service._cached_engine = engine_service.DEFAULT_ENGINE
    yield
    engine_service._cached_engine = engine_service.DEFAULT_ENGINE


def test_default_model_key_is_scoped_per_engine():
    assert settings_service.default_model_key("ollama") != settings_service.default_model_key("matricxon")


@pytest.mark.asyncio
async def test_get_default_model_falls_back_to_the_config_constant_when_nothing_set(db, user):
    from app.config import DEFAULT_MODEL

    assert await settings_service.get_default_model(db, user) == DEFAULT_MODEL


@pytest.mark.asyncio
async def test_set_default_model_then_get_round_trips(db, user):
    await settings_service.set_default_model(db, user.id, "my-tag")
    assert await settings_service.get_default_model(db, user) == "my-tag"


@pytest.mark.asyncio
async def test_default_model_survives_an_engine_switch_and_back(db, user):
    """The exact scenario reported live: set a default while Ollama is active, switch to Matricxon, switch
    back — the Ollama-scoped default must still be there, untouched, the whole time."""
    engine_service._cached_engine = "ollama"
    await settings_service.set_default_model(db, user.id, "ollama-tag")

    engine_service._cached_engine = "matricxon"
    assert await settings_service.get_default_model(db, user) != "ollama-tag"

    engine_service._cached_engine = "ollama"
    assert await settings_service.get_default_model(db, user) == "ollama-tag"


@pytest.mark.asyncio
async def test_each_engine_keeps_its_own_independent_default(db, user):
    engine_service._cached_engine = "ollama"
    await settings_service.set_default_model(db, user.id, "ollama-tag")
    engine_service._cached_engine = "matricxon"
    await settings_service.set_default_model(db, user.id, "matricxon-tag")

    engine_service._cached_engine = "ollama"
    assert await settings_service.get_default_model(db, user) == "ollama-tag"
    engine_service._cached_engine = "matricxon"
    assert await settings_service.get_default_model(db, user) == "matricxon-tag"


@pytest.mark.asyncio
async def test_get_default_model_falls_back_to_the_system_wide_row_for_this_engine(db, user):
    await settings_service.set_default_model(db, SYSTEM_OWNER_ID, "system-tag")
    assert await settings_service.get_default_model(db, user) == "system-tag"
