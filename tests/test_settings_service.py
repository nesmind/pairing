"""Tests for services/settings_service.py's proxy-mode and external-server-config getters/setters
(instance_pool.py uses the cached proxy-mode for routing). Title mode, channel delivery mode and the reply
timeout are in tests/test_chat_settings_service.py (moved to chat_settings_service.py for the file-size rule).
"""

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
    # Regression: local mode shows no Save button until something is installed, so saving "remote" with no hosts
    # left an admin with no UI path back to local (see OllamaServerConfig's validator).
    config = OllamaServerConfig(mode="remote", remote_hosts=[])
    assert config.mode == "local"


@pytest.mark.asyncio
async def test_set_ollama_server_config_normalizes_remote_with_no_hosts_to_local(db):
    await settings_service.set_ollama_server_config(db, OllamaServerConfig(mode="remote", remote_hosts=[]))

    config = await settings_service.get_ollama_server_config(db)

    assert config.mode == "local"


@pytest.mark.asyncio
async def test_get_ollama_server_config_self_heals_an_already_stuck_remote_row(db):
    # A row saved before this fix, stuck on "remote" with no hosts, must self-heal to "local" when read, not just on the
    # next save.
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
    """A row saved before mode/remote_hosts existed (the three original path fields) must load, with the new fields
    defaulted, not crash or drop an old value."""
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
    # See test_ollama_server_config_falls_back_to_local_when_remote_has_no_hosts: same regression and fix on
    # ComfyUIProcessConfig.
    config = ComfyUIProcessConfig(mode="remote", remote_hosts=[])
    assert config.mode == "local"


@pytest.mark.asyncio
async def test_set_comfyui_config_normalizes_remote_with_no_hosts_to_local(db):
    await settings_service.set_comfyui_config(db, ComfyUIProcessConfig(mode="remote", remote_hosts=[]))

    config = await settings_service.get_comfyui_config(db)

    assert config.mode == "local"


# default model (per-engine)
# Bug found: one shared "default_model" key meant switching engines had to wipe it (the removed
# engine_switch_service), losing the admin's preference. default_model_key scopes it per engine, so neither a switch nor
# a restart loses it.


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
    """The reported scenario: set a default under Ollama, switch to Matricxon and back; the Ollama-scoped default must
    stay untouched throughout."""
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
