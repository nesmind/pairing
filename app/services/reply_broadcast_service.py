"""In-process pub/sub hub for a conversation's in-flight assistant reply.

One topic per conversation_id, created lazily and kept for the process lifetime, never replaced. It holds the
current generation's snapshot (for late subscribers) and the subscriber queues watching. It serves
reply_generation_service's relay for the sender's tab (personal or channel) and routers/chat.py's
`/subscribe` for a channel's *other* members, or a personal owner reconnecting mid-reply from another
tab/device (see reply_generation_service's docstring for why one mechanism serves all).

Keeping one topic per conversation is load-bearing: a channel member's `/subscribe` is opened once and must
keep working across every message sent while they watch. A hub that swapped in a new topic (with an empty
subscriber set) on each start_topic orphaned connected subscribers at the second message: the first
displayed (a subscriber connecting *after* generation starts attaches fine), later ones showed nothing to a
member who had sat in the chat throughout. That was a shipped bug; a subscriber connecting *before* any
generation must still receive what comes next.

A module-level dict, not a class: one hub per process, like title_service's _background_title_tasks set.

Limitation: purely in-process memory, no Redis/broker. A subscriber whose request lands on a different
instance than the generation (instance_pool, with instance_count above 1) receives nothing. "Real" channel
delivery (chat_settings_service.get_channel_delivery_mode) is thus reliable only with one instance; "cheap"
(DB polling) has no such limit. A stated first-version limitation, not worked around with new infra.
"""

import asyncio
from dataclasses import dataclass, field

# Bounded so a lagging subscriber can't grow without limit: a full queue drops that one event (see publish)
# rather than block others or generation. Rare and harmless: the DB is authoritative and both the poller and
# the /subscribe client do a final reconcile fetch when a stream ends (chat.js).
_QUEUE_MAXSIZE = 1000


@dataclass
class GenerationState:
    """In-memory snapshot of a conversation's current or latest generation, so a mid-stream subscriber syncs at
    once instead of seeing a blank bubble. The DB (Message.content/status) is the durable truth.
    """

    message_id: str
    content: str = ""
    status: str = "streaming"
    sources: list[str] | None = None
    tool_events: list[dict] = field(default_factory=list)


class _Topic:
    __slots__ = ("state", "subscribers")

    def __init__(self) -> None:
        # None until the first generation calls start_topic; a subscriber connecting earlier still gets a live queue,
        # just nothing to sync.
        self.state: GenerationState | None = None
        self.subscribers: set[asyncio.Queue] = set()


# conversation_id -> _Topic, created on first subscribe or start_topic and never removed: cheap to keep, and
# removing it could delete it under a connected subscriber. Reusing one _Topic across messages (not replacing it) is the
# actual fix (see module docstring).
_topics: dict[str, _Topic] = {}


def _get_or_create_topic(conversation_id: str) -> _Topic:
    topic = _topics.get(conversation_id)
    if topic is None:
        topic = _Topic()
        _topics[conversation_id] = topic
    return topic


def start_topic(conversation_id: str, message_id: str) -> None:
    """Begins a new generation: resets the snapshot for `message_id` but never touches `subscribers`, so existing
    connections keep working and receive the new message's events.
    """
    _get_or_create_topic(conversation_id).state = GenerationState(message_id=message_id)


def _publish(conversation_id: str, event: dict) -> None:
    topic = _topics.get(conversation_id)
    if topic is None:
        return
    for queue in topic.subscribers:
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            # See _QUEUE_MAXSIZE: a dropped event for one lagging subscriber must never block the others.
            pass


def publish_chunk(conversation_id: str, chunk: str) -> None:
    topic = _topics.get(conversation_id)
    if topic is None or topic.state is None:
        return
    topic.state.content += chunk
    # "message_id" on every event: a /subscribe connection (any channel member but the sender) relays the hub's raw
    # events unmodified, so without it chat.js's handleReplyLiveEvent had nothing to key a bubble on and dropped
    # every chunk (a shipped bug). stream_reply's relay (the sender's tab) also stamps it: harmless, same value.
    _publish(conversation_id, {"chunk": chunk, "message_id": topic.state.message_id})


def publish_tool(conversation_id: str, tool_event: dict) -> None:
    topic = _topics.get(conversation_id)
    if topic is None or topic.state is None:
        return
    topic.state.tool_events.append(tool_event)
    _publish(conversation_id, {"tool": tool_event, "message_id": topic.state.message_id})


def publish_done(conversation_id: str, sources: list[str] | None) -> None:
    topic = _topics.get(conversation_id)
    if topic is None or topic.state is None:
        return
    topic.state.status = "complete"
    topic.state.sources = sources
    _publish(conversation_id, {"done": True, "sources": sources, "message_id": topic.state.message_id})


def publish_error(conversation_id: str, error: str) -> None:
    topic = _topics.get(conversation_id)
    if topic is None or topic.state is None:
        return
    topic.state.status = "error"
    _publish(conversation_id, {"error": error, "message_id": topic.state.message_id})


def publish_deleted(conversation_id: str) -> None:
    """Tells every live watcher (the sender's relay in stream_reply, other members' /subscribe) that the in-flight
    reply was deleted (conversation_service.delete_message, only caller). Like publish_error/publish_done,
    "message_id" included: it updates the snapshot a late subscriber syncs from, and "deleted" stops both relay
    loops like "done"/"error".
    """
    topic = _topics.get(conversation_id)
    if topic is None or topic.state is None:
        return
    topic.state.status = "deleted"
    _publish(conversation_id, {"deleted": True, "message_id": topic.state.message_id})


def subscribe(conversation_id: str) -> tuple[GenerationState | None, asyncio.Queue]:
    """Registers a subscriber, creating the topic if needed, and returns the current snapshot (None if no
    generation started) plus the queue for future events. Valid before any generation: the queue still gets the
    next start_topic/publish_* (the fix for the bug above). The snapshot is a shallow reference, safe to read
    once right after subscribing (single-threaded asyncio).
    """
    topic = _get_or_create_topic(conversation_id)
    queue: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
    topic.subscribers.add(queue)
    return topic.state, queue


def unsubscribe(conversation_id: str, queue: asyncio.Queue) -> None:
    topic = _topics.get(conversation_id)
    if topic is not None:
        topic.subscribers.discard(queue)
