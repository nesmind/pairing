"""Unit tests for app/services/channel_service.py (channel creation,
membership, and the manage-rights check that gates a channel
conversation's model/persona/rules/skill) plus the channel-aware
generalizations this feature made to conversation_service and
document_retrieval."""

import numpy as np
import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.database import Base
from app.models import Chunk, Document, User
from app.schemas import ChannelCreate, ChannelUpdate, ConversationUpdate
from app.services import channel_service, chat_attachment_service, conversation_service, document_retrieval
from app.services.auth_service import hash_password


@pytest.mark.asyncio
async def test_create_channel_builds_shared_conversation_and_membership(db, admin_user, user, channel_manager_user):
    body = ChannelCreate(
        name="general",
        member_user_ids=[user.id, channel_manager_user.id],
        manager_user_ids=[channel_manager_user.id],
    )
    channel = await channel_service.create_channel(db, body, admin_user)

    assert channel.conversation is not None
    assert channel.conversation.channel_id == channel.id
    assert channel.conversation.owner_id is None

    by_user = {m.user_id: m for m in channel.members}
    assert set(by_user) == {user.id, channel_manager_user.id}
    assert by_user[channel_manager_user.id].is_manager is True
    assert by_user[user.id].is_manager is False


@pytest.mark.asyncio
async def test_to_channel_out_skips_a_membership_whose_user_no_longer_exists(
    db, admin_user, user, channel_manager_user
):
    """The actual regression this covers: a ChannelMember row surviving its own User's deletion (confirmed live
    — user_service.delete_user didn't used to clean these up) made to_channel_out crash the *entire* admin
    Channels list over one bad row, since it read member.user.username unconditionally. member.user reads as
    None once its User is gone; to_channel_out must skip that membership, not blow up on it — while a real,
    still-existing member (channel_manager_user here) still shows up normally."""
    body = ChannelCreate(
        name="general", member_user_ids=[user.id, channel_manager_user.id], manager_user_ids=[channel_manager_user.id]
    )
    channel = await channel_service.create_channel(db, body, admin_user)

    # Simulates the orphaned row directly (rather than going through user_service.delete_user, which now cleans
    # this up itself) — this test's job is proving to_channel_out survives one existing regardless of how it
    # got there, not re-testing delete_user's own cleanup (see test_user_service.py for that).
    await db.delete(user)
    await db.commit()
    await db.refresh(channel)

    out = channel_service.to_channel_out(channel)

    assert [m.user_id for m in out.members] == [channel_manager_user.id]


@pytest.mark.asyncio
async def test_create_channel_rejects_manager_without_channel_manager_role(db, admin_user, user):
    body = ChannelCreate(name="general", member_user_ids=[user.id], manager_user_ids=[user.id])
    with pytest.raises(HTTPException) as exc_info:
        await channel_service.create_channel(db, body, admin_user)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_create_channel_rejects_manager_not_in_members(db, admin_user, user, channel_manager_user):
    body = ChannelCreate(name="general", member_user_ids=[user.id], manager_user_ids=[channel_manager_user.id])
    with pytest.raises(HTTPException) as exc_info:
        await channel_service.create_channel(db, body, admin_user)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_create_channel_rejects_zero_managers(db, admin_user, user):
    body = ChannelCreate(name="general", member_user_ids=[user.id], manager_user_ids=[])
    with pytest.raises(HTTPException) as exc_info:
        await channel_service.create_channel(db, body, admin_user)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_update_channel_rejects_dropping_to_zero_managers(db, admin_user, user, channel_manager_user):
    channel = await channel_service.create_channel(
        db,
        ChannelCreate(
            name="general",
            member_user_ids=[user.id, channel_manager_user.id],
            manager_user_ids=[channel_manager_user.id],
        ),
        admin_user,
    )
    with pytest.raises(HTTPException) as exc_info:
        await channel_service.update_channel(
            db,
            channel,
            ChannelUpdate(member_user_ids=[user.id, channel_manager_user.id], manager_user_ids=[]),
        )
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_update_channel_replaces_membership_wholesale(db, admin_user, user, channel_manager_user):
    channel = await channel_service.create_channel(
        db,
        ChannelCreate(
            name="general",
            member_user_ids=[user.id, channel_manager_user.id],
            manager_user_ids=[channel_manager_user.id],
        ),
        admin_user,
    )
    channel = await channel_service.update_channel(
        db,
        channel,
        ChannelUpdate(member_user_ids=[channel_manager_user.id], manager_user_ids=[channel_manager_user.id]),
    )
    by_user = {m.user_id: m for m in channel.members}
    assert set(by_user) == {channel_manager_user.id}
    assert by_user[channel_manager_user.id].is_manager is True


@pytest.mark.asyncio
async def test_update_channel_survives_a_fresh_session_after_create():
    """Regression test: PATCH /api/settings/channels/{id} used to 500
    with sqlalchemy.exc.MissingGreenlet. update_channel's blanket
    `await db.refresh(channel)` expired channel.members/conversation
    without safely reloading them under AsyncSession, so the very next
    touch — channel_service.to_channel_out, called by the router right
    after update_channel returns — attempted a synchronous lazy load and
    crashed. This only reproduces across two separate sessions (exactly
    like two separate HTTP requests: one to create the channel, a later
    one to edit it) — the shared single-session `db` fixture used by
    every other test in this file masks it completely, which is why this
    test builds its own second engine/session instead.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as setup_db:
        admin = User(username="admin", password_hash=hash_password("x"), role="admin")
        member = User(username="bob", password_hash=hash_password("x"), role="user")
        manager = User(username="carol", password_hash=hash_password("x"), role="channel_manager")
        setup_db.add_all([admin, member, manager])
        await setup_db.commit()
        admin_id, member_id, manager_id = admin.id, member.id, manager.id

    async with session_factory() as create_db:
        admin = await create_db.get(User, admin_id)
        channel = await channel_service.create_channel(
            create_db,
            ChannelCreate(
                name="general",
                member_user_ids=[member_id, manager_id],
                manager_user_ids=[manager_id],
            ),
            admin,
        )
        channel_id = channel.id

    # A fresh session, like a second HTTP request — must not raise.
    async with session_factory() as edit_db:
        channel = await channel_service.get_channel_or_404(edit_db, channel_id)
        channel = await channel_service.update_channel(
            edit_db,
            channel,
            ChannelUpdate(member_user_ids=[member_id, manager_id], manager_user_ids=[manager_id]),
        )
        out = channel_service.to_channel_out(channel)
        assert {m.username for m in out.members} == {"bob", "carol"}

    await engine.dispose()


@pytest.mark.asyncio
async def test_can_manage_channel_conversation(db, admin_user, user, channel_manager_user):
    channel = await channel_service.create_channel(
        db,
        ChannelCreate(
            name="general",
            member_user_ids=[user.id, channel_manager_user.id],
            manager_user_ids=[channel_manager_user.id],
        ),
        admin_user,
    )
    conversation = channel.conversation

    assert channel_service.can_manage_channel_conversation(conversation, admin_user) is True
    assert channel_service.can_manage_channel_conversation(conversation, channel_manager_user) is True
    assert channel_service.can_manage_channel_conversation(conversation, user) is False


@pytest.mark.asyncio
async def test_get_accessible_conversation_or_404_channel_membership(db, admin_user, user, channel_manager_user):
    # A *separate* channel_manager-role user from the channel_manager_user
    # fixture — that fixture is deliberately left out of this channel's
    # membership below, since the test needs a genuine non-member to
    # assert the 404 case against.
    manager = User(username="another_manager", password_hash=hash_password("x"), role="channel_manager")
    db.add(manager)
    await db.commit()

    channel = await channel_service.create_channel(
        db,
        ChannelCreate(name="general", member_user_ids=[user.id, manager.id], manager_user_ids=[manager.id]),
        admin_user,
    )
    conversation_id = channel.conversation.id

    found = await conversation_service.get_accessible_conversation_or_404(db, conversation_id, user)
    assert found.id == conversation_id

    with pytest.raises(HTTPException) as exc_info:
        await conversation_service.get_accessible_conversation_or_404(db, conversation_id, channel_manager_user)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_update_conversation_rejects_plain_members_model_change_on_channel(
    db,
    admin_user,
    user,
    channel_manager_user,
    monkeypatch,
):
    channel = await channel_service.create_channel(
        db,
        ChannelCreate(
            name="general",
            member_user_ids=[user.id, channel_manager_user.id],
            manager_user_ids=[channel_manager_user.id],
        ),
        admin_user,
    )
    conversation = channel.conversation

    async def fake_installed():
        return ["llama3:latest"]

    monkeypatch.setattr(conversation_service, "installed_chat_models", fake_installed)

    with pytest.raises(HTTPException) as exc_info:
        await conversation_service.update_conversation(
            db,
            conversation,
            ConversationUpdate(model="llama3:latest"),
            user,
        )
    assert exc_info.value.status_code == 403

    updated = await conversation_service.update_conversation(
        db,
        conversation,
        ConversationUpdate(model="llama3:latest"),
        channel_manager_user,
    )
    assert updated.model == "llama3:latest"


@pytest.mark.asyncio
async def test_update_conversation_rejects_title_change_on_channel(db, admin_user, user, channel_manager_user):
    channel = await channel_service.create_channel(
        db,
        ChannelCreate(
            name="general",
            member_user_ids=[user.id, channel_manager_user.id],
            manager_user_ids=[channel_manager_user.id],
        ),
        admin_user,
    )
    with pytest.raises(HTTPException) as exc_info:
        await conversation_service.update_conversation(
            db,
            channel.conversation,
            ConversationUpdate(title="Renamed"),
            admin_user,
        )
    assert exc_info.value.status_code == 400


def _document_with_chunk(*, owner_id, filename, vector):
    document = Document(owner_id=owner_id, filename=filename, content_hash="x" * 64, chunk_count=1, size_bytes=1)
    chunk = Chunk(document=document, text="chunk text", embedding=np.asarray(vector, dtype=np.float32).tobytes())
    return document, chunk


@pytest.mark.asyncio
async def test_retrieve_context_channel_scope_excludes_private_docs(db, user, monkeypatch):
    global_doc, global_chunk = _document_with_chunk(owner_id=None, filename="global/handbook.txt", vector=[1.0, 0.0])
    private_doc, private_chunk = _document_with_chunk(
        owner_id=user.id,
        filename=f"users/{user.id}/notes.txt",
        vector=[1.0, 0.0],
    )
    db.add_all([global_doc, private_doc, global_chunk, private_chunk])
    await db.commit()

    async def fake_embed(_query):
        return [1.0, 0.0]

    monkeypatch.setattr(document_retrieval, "embed", fake_embed)

    channel_scoped = await document_retrieval.retrieve_context(db, "question", user_id=None)
    assert [r["filename"] for r in channel_scoped] == ["handbook.txt"]

    personal_scoped = await document_retrieval.retrieve_context(db, "question", user_id=user.id)
    assert {r["filename"] for r in personal_scoped} == {"handbook.txt", "notes.txt"}


@pytest.mark.asyncio
async def test_delete_channel_removes_its_conversations_attachments_folder(
    db, admin_user, user, channel_manager_user, tmp_path, monkeypatch
):
    """Deleting a channel takes its shared conversation (and every
    message in it) down via the ORM cascade — see
    chat_attachment_service.delete_conversation_attachments, called from
    delete_channel before that cascade runs, since the cascade itself has
    no idea attachment files exist on disk at all."""
    monkeypatch.setattr(chat_attachment_service, "ATTACHMENTS_DIR", tmp_path)

    channel = await channel_service.create_channel(
        db,
        ChannelCreate(
            name="general",
            member_user_ids=[user.id, channel_manager_user.id],
            manager_user_ids=[channel_manager_user.id],
        ),
        admin_user,
    )
    attachment_dir = tmp_path / channel.conversation.id
    attachment_dir.mkdir()
    (attachment_dir / "abc123_photo.png").write_bytes(b"fake")

    await channel_service.delete_channel(db, channel)

    assert not attachment_dir.exists()
