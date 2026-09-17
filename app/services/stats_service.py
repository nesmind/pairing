"""Usage-stats aggregation for the Stats page's default dashboard view
(GET /api/stats/summary, see app/routers/stats.py). Every number here is
computed with a SQL-level func.count()/group_by() query — never a full
table scan pulled into Python (see app/services/user_service.py's
active_admin_count for the existing count-query precedent this mirrors)
— since messages/chunks can get large in a long-running deployment.
Covers only what's already modeled (see app/models/__init__.py): no new
tracking/analytics tables. A conversation with model=None (never
started) is excluded from by_model rather than shown as a fake "(none)"
bucket.
"""

from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Conversation, Document, Message, User
from app.models._base import utcnow
from app.schemas import (
    ConversationStats,
    CountByKey,
    KnowledgeBaseStats,
    LiveConversationStats,
    MessageStats,
    StatsSummary,
    UserStats,
)

# How far back a message's own updated_at (see app/models/conversation.py's
# Message.updated_at doc comment — bumped on every streamed-content flush,
# not just on creation) still counts as "posting last minute" for the Stats
# page's "Live conversations" section below.
_RECENTLY_ACTIVE_WINDOW = timedelta(minutes=5)


async def _conversation_stats(db: AsyncSession) -> ConversationStats:
    total = (await db.execute(select(func.count(Conversation.id)))).scalar_one()
    personal = (
        await db.execute(select(func.count(Conversation.id)).where(Conversation.owner_id.isnot(None)))
    ).scalar_one()
    channel = (
        await db.execute(select(func.count(Conversation.id)).where(Conversation.channel_id.isnot(None)))
    ).scalar_one()
    model_rows = (
        await db.execute(
            select(Conversation.model, func.count(Conversation.id))
            .where(Conversation.model.isnot(None))
            .group_by(Conversation.model)
            .order_by(func.count(Conversation.id).desc())
        )
    ).all()
    return ConversationStats(
        total=total,
        personal=personal,
        channel=channel,
        by_model=[CountByKey(key=model, count=count) for model, count in model_rows],
    )


async def _message_stats(db: AsyncSession) -> MessageStats:
    total = (await db.execute(select(func.count(Message.id)))).scalar_one()
    role_rows = (await db.execute(select(Message.role, func.count(Message.id)).group_by(Message.role))).all()
    status_rows = (await db.execute(select(Message.status, func.count(Message.id)).group_by(Message.status))).all()
    return MessageStats(
        total=total,
        by_role=[CountByKey(key=role, count=count) for role, count in role_rows],
        by_status=[CountByKey(key=status, count=count) for status, count in status_rows],
    )


async def _user_stats(db: AsyncSession) -> UserStats:
    total = (await db.execute(select(func.count(User.id)))).scalar_one()
    active = (await db.execute(select(func.count(User.id)).where(User.status == "active"))).scalar_one()
    role_rows = (await db.execute(select(User.role, func.count(User.id)).group_by(User.role))).all()
    return UserStats(
        total=total,
        active=active,
        disabled=total - active,
        by_role=[CountByKey(key=role, count=count) for role, count in role_rows],
    )


async def _knowledge_base_stats(db: AsyncSession) -> KnowledgeBaseStats:
    total_documents = (await db.execute(select(func.count(Document.id)))).scalar_one()
    total_chunks = (await db.execute(select(func.count(Chunk.id)))).scalar_one()
    global_documents = (
        await db.execute(select(func.count(Document.id)).where(Document.owner_id.is_(None)))
    ).scalar_one()
    return KnowledgeBaseStats(
        total_documents=total_documents,
        total_chunks=total_chunks,
        global_documents=global_documents,
        private_documents=total_documents - global_documents,
    )


async def _live_conversation_stats(db: AsyncSession) -> LiveConversationStats:
    """Live here means only what's actually observable server-side — no
    per-viewer presence/scrolling tracking exists anywhere in this app.
    streaming_now = a reply is literally mid-generation right now
    (Message.status == "streaming", i.e. "waiting for model reply").
    recently_active = some message in the conversation changed within
    _RECENTLY_ACTIVE_WINDOW (Message.updated_at, which is bumped on every
    streamed-content flush too — see its own doc comment — so this also
    captures "posting last minute" without needing a separate signal).
    total is the union of both sets, not their sum, since a conversation
    can satisfy both at once."""
    streaming_ids = (
        (await db.execute(select(Message.conversation_id).where(Message.status == "streaming").distinct()))
        .scalars()
        .all()
    )
    recent_ids = (
        (
            await db.execute(
                select(Message.conversation_id)
                .where(Message.updated_at >= utcnow() - _RECENTLY_ACTIVE_WINDOW)
                .distinct()
            )
        )
        .scalars()
        .all()
    )
    streaming_set = set(streaming_ids)
    recent_set = set(recent_ids)
    return LiveConversationStats(
        total=len(streaming_set | recent_set),
        streaming_now=len(streaming_set),
        recently_active=len(recent_set),
    )


async def get_stats_summary(db: AsyncSession) -> StatsSummary:
    """The one entry point app/routers/stats.py calls — assembles every
    section above."""
    return StatsSummary(
        conversations=await _conversation_stats(db),
        messages=await _message_stats(db),
        users=await _user_stats(db),
        knowledge_base=await _knowledge_base_stats(db),
        live_conversations=await _live_conversation_stats(db),
    )
