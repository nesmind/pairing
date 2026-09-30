"""POST /api/settings/delete-model, called directly (see tests/test_ollama_admin_router.py's docstring)."""

import pytest

from app.routers import settings as settings_router
from app.schemas import PullModelRequest
from app.services.extended_model_catalog_service import ExtendedModelCatalog


async def _noop_delete(_tag: str) -> None:
    return None


@pytest.mark.asyncio
async def test_uninstalling_removes_the_model_from_browse_more_models(db, monkeypatch):
    """Whether it was added by search or found installed, an uninstalled model must not return as a suggestion."""
    monkeypatch.setattr(settings_router, "delete_model", _noop_delete)
    catalog = ExtendedModelCatalog(db)
    await catalog._save([{"tag": "hf.co/a/b:c", "note": None}, {"tag": "hf.co/x/y:z", "note": None}])

    result = await settings_router.delete_model_endpoint(PullModelRequest(tag="hf.co/a/b:c"), db=db, _admin=None)

    assert result.deleted == "hf.co/a/b:c"
    assert [e["tag"] for e in await catalog.list()] == ["hf.co/x/y:z"]
