"""A disabled (hidden) model is refused for regular users, but not for admins."""

from types import SimpleNamespace

from app.services.model_catalog_service import HiddenModelTags


async def test_disabled_model_blocks_only_non_admins(db) -> None:
    tags = HiddenModelTags(db)
    await tags.set({"m1"})
    user, admin = SimpleNamespace(role="user"), SimpleNamespace(role="admin")

    assert await tags.blocks(user, "m1") is True
    assert await tags.blocks(user, "m2") is False
    assert await tags.blocks(admin, "m1") is False

    await tags.set(set())
    assert await tags.blocks(user, "m1") is False
