"""Response shapes for GET /api/stats/summary — the Stats page's default
usage-dashboard view (see app/services/stats_service.py for the actual
aggregation queries and app/static/js/stats.js for how these render)."""

from pydantic import BaseModel


class CountByKey(BaseModel):
    """One labeled count — the shape every "breakdown by X" section in
    the dashboard uses (model usage, message role/status, user role)
    rather than a bespoke schema per breakdown."""

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
    """No presence/heartbeat tracking exists in this app (see
    app/services/stats_service.py), so "live" is defined only from what's
    actually observable server-side: a reply mid-generation, or a message
    that changed very recently."""

    total: int
    streaming_now: int
    recently_active: int


class StatsSummary(BaseModel):
    conversations: ConversationStats
    messages: MessageStats
    users: UserStats
    knowledge_base: KnowledgeBaseStats
    live_conversations: LiveConversationStats
