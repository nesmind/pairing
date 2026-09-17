"""Unit tests for app/services/http_proxy_service.py and
app/schemas/http_proxy_config.py's HttpProxyConfig.proxy_url() — mirrors
tests/test_settings_service.py's own get_comfyui_config/set_comfyui_config
round-trip convention."""

import pytest

from app.models import SYSTEM_OWNER_ID, AppSetting
from app.schemas import HttpProxyConfig
from app.services import http_proxy_service


def test_proxy_url_is_none_when_disabled():
    config = HttpProxyConfig(enabled=False, host="10.0.0.5", port=8080)
    assert config.proxy_url() is None


def test_proxy_url_is_none_when_enabled_but_host_missing():
    config = HttpProxyConfig(enabled=True, host=None, port=8080)
    assert config.proxy_url() is None


def test_proxy_url_is_none_when_enabled_but_port_missing():
    config = HttpProxyConfig(enabled=True, host="10.0.0.5", port=None)
    assert config.proxy_url() is None


def test_proxy_url_builds_the_expected_url_when_fully_configured():
    config = HttpProxyConfig(enabled=True, host="10.0.0.5", port=8080)
    assert config.proxy_url() == "http://10.0.0.5:8080"


def test_proxy_url_includes_username_when_no_password_set():
    config = HttpProxyConfig(enabled=True, host="10.0.0.5", port=8080, username="alice")
    assert config.proxy_url() == "http://alice@10.0.0.5:8080"


def test_proxy_url_includes_username_and_password():
    config = HttpProxyConfig(enabled=True, host="10.0.0.5", port=8080, username="alice", password="s3cret")
    assert config.proxy_url() == "http://alice:s3cret@10.0.0.5:8080"


def test_proxy_url_percent_encodes_special_characters_in_credentials():
    config = HttpProxyConfig(enabled=True, host="10.0.0.5", port=8080, username="a@b", password="p:w/d")
    assert config.proxy_url() == "http://a%40b:p%3Aw%2Fd@10.0.0.5:8080"


def test_proxy_url_ignores_a_password_with_no_username():
    # A password with no username to pair it with isn't a valid "user:pass"
    # auth segment — proxy_url() only ever includes credentials when
    # there's a username to anchor them to.
    config = HttpProxyConfig(enabled=True, host="10.0.0.5", port=8080, password="orphaned")
    assert config.proxy_url() == "http://10.0.0.5:8080"


@pytest.mark.asyncio
async def test_get_http_proxy_config_defaults_to_disabled_when_nothing_saved(db):
    config = await http_proxy_service.get_http_proxy_config(db)
    assert config.enabled is False
    assert config.host is None
    assert config.port is None


@pytest.mark.asyncio
async def test_set_http_proxy_config_round_trips(db):
    await http_proxy_service.set_http_proxy_config(db, HttpProxyConfig(enabled=True, host="1.2.3.4", port=3128))

    config = await http_proxy_service.get_http_proxy_config(db)

    assert config.enabled is True
    assert config.host == "1.2.3.4"
    assert config.port == 3128


@pytest.mark.asyncio
async def test_set_http_proxy_config_overwrites_a_previously_saved_value(db):
    await http_proxy_service.set_http_proxy_config(db, HttpProxyConfig(enabled=True, host="1.2.3.4", port=3128))
    await http_proxy_service.set_http_proxy_config(db, HttpProxyConfig(enabled=False))

    config = await http_proxy_service.get_http_proxy_config(db)

    assert config.enabled is False
    assert config.host is None


@pytest.mark.asyncio
async def test_get_http_proxy_config_for_display_redacts_the_password(db):
    await http_proxy_service.set_http_proxy_config(
        db, HttpProxyConfig(enabled=True, host="1.2.3.4", port=3128, username="alice", password="s3cret")
    )

    displayed = await http_proxy_service.get_http_proxy_config_for_display(db)

    assert displayed.password is None
    assert displayed.has_password is True
    assert displayed.username == "alice"  # username isn't a secret — still shown


@pytest.mark.asyncio
async def test_get_http_proxy_config_for_display_reports_no_password_when_none_saved(db):
    await http_proxy_service.set_http_proxy_config(db, HttpProxyConfig(enabled=True, host="1.2.3.4", port=3128))

    displayed = await http_proxy_service.get_http_proxy_config_for_display(db)

    assert displayed.has_password is False


@pytest.mark.asyncio
async def test_set_http_proxy_config_with_a_blank_password_keeps_the_existing_one(db):
    await http_proxy_service.set_http_proxy_config(
        db, HttpProxyConfig(enabled=True, host="1.2.3.4", port=3128, password="original")
    )
    # Simulates the admin saving again (e.g. just flipping "enabled")
    # without retyping a password they were never shown back.
    await http_proxy_service.set_http_proxy_config(db, HttpProxyConfig(enabled=False, host="1.2.3.4", port=3128))

    config = await http_proxy_service.get_http_proxy_config(db)

    assert config.password == "original"


@pytest.mark.asyncio
async def test_set_http_proxy_config_with_a_new_password_replaces_the_existing_one(db):
    await http_proxy_service.set_http_proxy_config(
        db, HttpProxyConfig(enabled=True, host="1.2.3.4", port=3128, password="original")
    )
    await http_proxy_service.set_http_proxy_config(
        db, HttpProxyConfig(enabled=True, host="1.2.3.4", port=3128, password="replaced")
    )

    config = await http_proxy_service.get_http_proxy_config(db)

    assert config.password == "replaced"


@pytest.mark.asyncio
async def test_set_http_proxy_config_stores_the_password_encrypted_not_plaintext(db):
    await http_proxy_service.set_http_proxy_config(
        db, HttpProxyConfig(enabled=True, host="1.2.3.4", port=3128, password="s3cret")
    )

    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, http_proxy_service.HTTP_PROXY_CONFIG_KEY))

    assert "password" not in row.value
    assert row.value["password_encrypted"] != "s3cret"
    assert "s3cret" not in row.value["password_encrypted"]


@pytest.mark.asyncio
async def test_get_http_proxy_config_reads_a_legacy_plaintext_password(db):
    # A row saved before encryption was added would still carry a plain
    # "password" key rather than "password_encrypted" — must still read
    # back correctly rather than erroring or silently losing it.
    db.add(
        AppSetting(
            owner_id=SYSTEM_OWNER_ID,
            key=http_proxy_service.HTTP_PROXY_CONFIG_KEY,
            value={"enabled": True, "host": "1.2.3.4", "port": 3128, "password": "legacy-plaintext"},
        )
    )
    await db.commit()

    config = await http_proxy_service.get_http_proxy_config(db)

    assert config.password == "legacy-plaintext"
