"""The Models tab's download sections start hidden on a connector engine (routers/pages.py `remote_models`), so they
don't show for a moment before settings.js hides them."""

import re

import pytest
from starlette.requests import Request

from app.routers import pages
from app.services import engine_service

_SECTIONS = (
    "browse-more-models-section",
    "recommended-models-heading",
    "embedding-models-section",
    "diffusion-models-section",
)


@pytest.fixture(autouse=True)
def reset_engine():
    engine_service._cached_engine = engine_service.DEFAULT_ENGINE
    yield
    engine_service._cached_engine = engine_service.DEFAULT_ENGINE


def _request(user_id: str) -> Request:
    scope = {"type": "http", "method": "GET", "path": "/admin-settings", "headers": [], "query_string": b""}
    return Request({**scope, "session": {"user_id": user_id}})


async def _hidden_sections(admin_user, db) -> set[str]:
    html = (await pages.admin_settings_page(_request(admin_user.id), db=db)).body.decode()
    hidden = set()
    for section in _SECTIONS:
        classes = re.search(rf'id="{section}" class="([^"]*)"', html).group(1).split()
        if "hidden" in classes:
            hidden.add(section)
    return hidden


@pytest.mark.asyncio
async def test_sections_start_hidden_on_a_connector_engine(db, admin_user):
    engine_service._cached_engine = "cloudflare"
    assert await _hidden_sections(admin_user, db) == set(_SECTIONS)


@pytest.mark.asyncio
async def test_sections_start_visible_on_a_local_engine(db, admin_user):
    engine_service._cached_engine = "matricxon"
    assert await _hidden_sections(admin_user, db) == set()
