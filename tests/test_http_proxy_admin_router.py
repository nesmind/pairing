"""Unit tests for app/routers/http_proxy_admin.py — called directly, not
through a TestClient/ASGI app (see tests/test_instance_proxy_http.py's
own docstring on why this project doesn't use that pattern)."""

import pytest

from app.routers import http_proxy_admin
from app.schemas import HttpProxyConfig


@pytest.mark.asyncio
async def test_get_http_proxy_defaults_to_disabled(db):
    config = await http_proxy_admin.get_http_proxy(db=db, _admin=None)
    assert config.enabled is False


@pytest.mark.asyncio
async def test_update_http_proxy_persists_and_round_trips(db):
    body = HttpProxyConfig(enabled=True, host="10.0.0.5", port=8080)

    result = await http_proxy_admin.update_http_proxy(body, db=db, _admin=None)

    assert result.enabled is True
    assert result.host == "10.0.0.5"
    assert result.port == 8080
    assert (await http_proxy_admin.get_http_proxy(db=db, _admin=None)).port == 8080


@pytest.mark.asyncio
async def test_update_http_proxy_never_echoes_the_real_password_back(db):
    body = HttpProxyConfig(enabled=True, host="10.0.0.5", port=8080, username="alice", password="s3cret")

    result = await http_proxy_admin.update_http_proxy(body, db=db, _admin=None)

    assert result.password is None
    assert result.has_password is True


@pytest.mark.asyncio
async def test_get_http_proxy_never_returns_the_real_password(db):
    await http_proxy_admin.update_http_proxy(
        HttpProxyConfig(enabled=True, host="10.0.0.5", port=8080, password="s3cret"), db=db, _admin=None
    )

    result = await http_proxy_admin.get_http_proxy(db=db, _admin=None)

    assert result.password is None
    assert result.has_password is True
