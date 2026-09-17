"""Unit tests for app/services/user_service.py: first_name/last_name
handling — in particular the "empty string clears it, omitted leaves it
alone" behavior update_user relies on (see its "or None" comment) — and
delete_user's own cascade cleanup."""

import pytest
from sqlalchemy import select

from app.models import ChannelMember
from app.schemas import ChannelCreate, UserCreate, UserUpdate
from app.services import channel_service, user_service


@pytest.mark.asyncio
async def test_create_user_stores_and_normalizes_names(db, monkeypatch):
    async def fake_installed():
        return ["llama3:latest"]

    async def fake_default_model(_db, _installed):
        return "llama3:latest"

    monkeypatch.setattr(user_service, "installed_chat_models", fake_installed)
    monkeypatch.setattr(user_service, "get_default_model_for_new_users", fake_default_model)

    with_name = await user_service.create_user(
        db,
        UserCreate(username="alice", password="alice-pw", first_name="Alice", last_name="Smith"),
    )
    assert with_name.first_name == "Alice"
    assert with_name.last_name == "Smith"

    # An empty string (not just an omitted field) normalizes down to
    # NULL rather than being stored as a blank string.
    blank_name = await user_service.create_user(
        db,
        UserCreate(username="bob", password="bob-password", first_name="", last_name=""),
    )
    assert blank_name.first_name is None
    assert blank_name.last_name is None


@pytest.mark.asyncio
async def test_update_user_can_set_then_clear_a_name(db, monkeypatch):
    async def fake_installed():
        return ["llama3:latest"]

    async def fake_default_model(_db, _installed):
        return "llama3:latest"

    monkeypatch.setattr(user_service, "installed_chat_models", fake_installed)
    monkeypatch.setattr(user_service, "get_default_model_for_new_users", fake_default_model)

    await user_service.create_user(db, UserCreate(username="carol", password="carol-pw"))

    updated = await user_service.update_user(db, "carol", UserUpdate(first_name="Carol", last_name="Jones"))
    assert updated.first_name == "Carol"
    assert updated.last_name == "Jones"

    # Explicitly sending "" (as the Settings UI always does — see
    # settings.js's save handler) clears a previously-set name; omitting
    # the field entirely (not exercised here) would leave it untouched.
    cleared = await user_service.update_user(db, "carol", UserUpdate(first_name="", last_name=""))
    assert cleared.first_name is None
    assert cleared.last_name is None


@pytest.mark.asyncio
async def test_delete_user_also_removes_their_channel_memberships(db, admin_user, user, channel_manager_user):
    """The regression this covers: a ChannelMember row surviving its own User's deletion crashed the entire
    admin Channels list over one bad row (confirmed live) — channel_service.to_channel_out reads
    member.user.username unconditionally, and a membership pointing at a deleted user has no User to read that
    from. delete_user must clean these up the same way it already does for the user's own conversations/
    documents/notes."""
    body = ChannelCreate(
        name="general", member_user_ids=[user.id, channel_manager_user.id], manager_user_ids=[channel_manager_user.id]
    )
    await channel_service.create_channel(db, body, admin_user)

    await user_service.delete_user(db, user.username)

    remaining = (await db.execute(select(ChannelMember).where(ChannelMember.user_id == user.id))).scalars().all()
    assert remaining == []
