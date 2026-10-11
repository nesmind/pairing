"""Response shapes for GET /api/stats/summary, the Stats dashboard (queries: services/stats_service.py;
rendering: static/js/stats.js).
"""

from pydantic import BaseModel


class CountByKey(BaseModel):
    """One labeled count, shared by the dashboard's "breakdown by X" sections (model usage, message role/status, user
    role)."""

    key: str
    count: int


class ConversationStats(BaseModel):
    total: int
    personal: int
    channel: int
    by_model: list[CountByKey]


class MessageStats(BaseModel):
    total: int
    by_role: list[CountByKey]
    by_status: list[CountByKey]


class UserStats(BaseModel):
    total: int
    active: int
    disabled: int
    by_role: list[CountByKey]


class KnowledgeBaseStats(BaseModel):
    total_documents: int
    total_chunks: int
    global_documents: int
    private_documents: int


class LiveConversationStats(BaseModel):
    """There is no presence tracking (stats_service.py), so "live" means observable server-side: a reply
    mid-generation or a message changed very recently.
    """

    total: int
    streaming_now: int
    recently_active: int


class StatsSummary(BaseModel):
    conversations: ConversationStats
    messages: MessageStats
    users: UserStats
    knowledge_base: KnowledgeBaseStats
    live_conversations: LiveConversationStats
