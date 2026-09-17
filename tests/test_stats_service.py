"""Unit tests for app/services/stats_service.py — every number is a real
SQL-level count/group_by, so these tests build a handful of rows
directly and assert the aggregation, rather than exercising any other
service's own creation logic."""

from datetime import timedelta

import pytest

from app.models import Channel, Chunk, Conversation, Document, Message, User
from app.models._base import utcnow
from app.services import stats_service
from app.services.auth_service import hash_password


def _by_key(items, key):
    return next((entry.count for entry in items if entry.key == key), None)


@pytest.mark.asyncio
async def test_get_stats_summary_on_an_empty_database(db):
    summary = await stats_service.get_stats_summary(db)

    assert summary.conversations.total == 0
    assert summary.conversations.by_model == []
    assert summary.messages.total == 0
    assert summary.messages.by_role == []
    assert summary.users.total == 0
    assert summary.knowledge_base.total_documents == 0
    assert summary.live_conversations.total == 0


@pytest.mark.asyncio
async def test_conversation_stats_splits_personal_and_channel_and_groups_by_model(db, user):
    personal = Conversation(owner_id=user.id, model="llama3:latest")
    also_llama = Conversation(owner_id=user.id, model="llama3:latest")
    never_started = Conversation(owner_id=user.id, model=None)
    db.add_all([personal, also_llama, never_started])

    channel = Channel(name="general", conversation=Conversation(model="gemma:latest"))
    db.add(channel)
    await db.commit()

    summary = await stats_service.get_stats_summary(db)

    assert summary.conversations.total == 4
    assert summary.conversations.personal == 3
    assert summary.conversations.channel == 1
    # Two conversations sharing a model aggregate into one entry with
    # count 2, not two separate entries — and the never-started one
    # (model=None) is excluded entirely rather than shown as a fake bucket.
    assert _by_key(summary.conversations.by_model, "llama3:latest") == 2
    assert _by_key(summary.conversations.by_model, "gemma:latest") == 1
    assert sum(entry.count for entry in summary.conversations.by_model) == 3


@pytest.mark.asyncio
async def test_message_stats_breaks_down_by_role_and_status(db, user):
    conversation = Conversation(owner_id=user.id, model="llama3:latest")
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)

    db.add_all(
        [
            Message(conversation_id=conversation.id, role="user", content="hi", status="complete"),
            Message(conversation_id=conversation.id, role="assistant", content="hello", status="complete"),
            Message(conversation_id=conversation.id, role="assistant", content="", status="error"),
        ]
    )
    await db.commit()

    summary = await stats_service.get_stats_summary(db)

    assert summary.messages.total == 3
    assert _by_key(summary.messages.by_role, "user") == 1
    assert _by_key(summary.messages.by_role, "assistant") == 2
    assert _by_key(summary.messages.by_status, "complete") == 2
    assert _by_key(summary.messages.by_status, "error") == 1


@pytest.mark.asyncio
async def test_user_stats_counts_active_disabled_and_role(db, user, admin_user):
    disabled = User(username="gone", password_hash=hash_password("x"), role="user", status="disabled")
    db.add(disabled)
    await db.commit()

    summary = await stats_service.get_stats_summary(db)

    # `user`/`admin_user` fixtures are both active by default.
    assert summary.users.total == 3
    assert summary.users.active == 2
    assert summary.users.disabled == 1
    assert _by_key(summary.users.by_role, "admin") == 1
    assert _by_key(summary.users.by_role, "user") == 2


@pytest.mark.asyncio
async def test_knowledge_base_stats_splits_global_and_private_documents(db, user):
    global_doc = Document(owner_id=None, filename="global/handbook.pdf", content_hash="a" * 64, chunk_count=2)
    private_doc = Document(owner_id=user.id, filename=f"users/{user.id}/notes.txt", content_hash="b" * 64)
    db.add_all([global_doc, private_doc])
    await db.commit()
    await db.refresh(global_doc)

    db.add_all(
        [
            Chunk(document_id=global_doc.id, text="a", embedding=b"fake"),
            Chunk(document_id=global_doc.id, text="b", embedding=b"fake"),
        ]
    )
    await db.commit()

    summary = await stats_service.get_stats_summary(db)

    assert summary.knowledge_base.total_documents == 2
    assert summary.knowledge_base.global_documents == 1
    assert summary.knowledge_base.private_documents == 1
    assert summary.knowledge_base.total_chunks == 2


@pytest.mark.asyncio
async def test_live_conversation_stats_distinguishes_streaming_and_recent_and_stale(db, user):
    streaming_conv = Conversation(owner_id=user.id, model="llama3:latest")
    recent_conv = Conversation(owner_id=user.id, model="llama3:latest")
    stale_conv = Conversation(owner_id=user.id, model="llama3:latest")
    db.add_all([streaming_conv, recent_conv, stale_conv])
    await db.commit()
    for conv in (streaming_conv, recent_conv, stale_conv):
        await db.refresh(conv)

    db.add_all(
        [
            Message(conversation_id=streaming_conv.id, role="assistant", content="", status="streaming"),
            Message(conversation_id=recent_conv.id, role="user", content="hi", status="complete"),
            Message(
                conversation_id=stale_conv.id,
                role="user",
                content="hi, a while ago",
                status="complete",
                updated_at=utcnow() - timedelta(hours=1),
            ),
        ]
    )
    await db.commit()

    summary = await stats_service.get_stats_summary(db)

    # streaming_conv and recent_conv both have a message updated just now,
    # so both land in recently_active — streaming_conv also counts toward
    # streaming_now. stale_conv's message falls outside the window and
    # isn't counted anywhere. total is the union, not the sum.
    assert summary.live_conversations.streaming_now == 1
    assert summary.live_conversations.recently_active == 2
    assert summary.live_conversations.total == 2
