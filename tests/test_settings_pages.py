"""/settings (user level, every account) and /admin-settings (admins only), the same template."""

import re
from pathlib import Path

import pytest
from fastapi.responses import RedirectResponse
from starlette.requests import Request

from app.routers import pages


def _request(user_id: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
            "query_string": b"",
            "session": {"user_id": user_id},
        }
    )


def _tabs(response) -> list[str]:
    return re.findall(r'data-tab="([^"]+)"', response.body.decode())


@pytest.mark.asyncio
async def test_user_settings_has_only_user_tabs_even_for_an_admin(db, user, admin_user):
    for who in (user, admin_user):
        response = await pages.settings_page(_request(who.id), db=db)
        html = response.body.decode()
        assert _tabs(response) == ["model", "behavior", "knowledge", "account"]
        assert "will not affect other users" in html
        assert 'id="hf-search-input"' not in html and 'id="mcp-panel"' not in html
        for admin_only in ("tab-system", "tab-external-servers", "tab-mcp-servers", "tab-users", "tab-channels"):
            assert f'id="{admin_only}"' not in html
        assert ('href="/admin-settings"' in html) is (who is admin_user)


@pytest.mark.asyncio
async def test_admin_settings_has_the_admin_tabs_and_the_admin_models_page(db, admin_user):
    response = await pages.admin_settings_page(_request(admin_user.id), db=db)
    html = response.body.decode()

    assert _tabs(response) == ["model", "system", "external-servers", "mcp-servers", "users", "channels"]
    assert 'id="hf-search-input"' in html and 'id="mcp-panel"' in html
    assert "will not affect other users" not in html
    # truly separate: none of the user-level panels or controls are in this page
    for user_only in ("tab-behavior", "tab-knowledge", "tab-account", "save-bar", "rag-top-k", "account-timezone"):
        assert f'id="{user_only}"' not in html


@pytest.mark.asyncio
async def test_a_regular_user_is_sent_back_from_admin_settings(db, user):
    response = await pages.admin_settings_page(_request(user.id), db=db)

    assert isinstance(response, RedirectResponse) and response.headers["location"] == "/settings"


@pytest.mark.asyncio
async def test_matricxon_form_has_the_cache_folder_field_and_the_script_keeps_cluster_role(db, admin_user):
    html = (await pages.admin_settings_page(_request(admin_user.id), db=db)).body.decode()
    assert 'data-field="prompt_cache_path"' in html
    script = (Path(__file__).parent.parent / "app" / "static" / "js" / "settings.js").read_text()
    assert "prompt_cache_path:" in script and "cluster_role: section.dataset.clusterRole" in script
    assert 'getElementById("matricxon-cache-panel")' in script
