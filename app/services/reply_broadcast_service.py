"""In-process pub/sub hub for a conversation's in-flight assistant reply.

One topic per conversation_id — created lazily on first use and kept for
the life of the process, never replaced — holding the current
generation's snapshot (for a late subscriber to sync against) plus the
set of subscriber queues currently watching this conversation. Serves
every caller that cares about a conversation's live generation state:
app.services.reply_generation_service's own relay for the sender's tab
(personal or channel alike), and app/routers/chat.py's `/subscribe`
endpoint for a channel's *other* members, or a personal chat's owner
reconnecting mid-reply from a different tab/device after a disconnect —
see reply_generation_service's module docstring for why unifying all of
these into one mechanism, rather than building "real" delivery mode as a
separate add-on, is what keeps every caller simple.

Keeping one topic per conversation, rather than replacing it with a
fresh one on every new message, is load-bearing, not just tidy: a
channel member's `/subscribe` connection is opened once (when they open
the chat) and expected to keep working for as long as they stay there,
across however many messages get sent while they're watching. A version
of this hub that swapped in a brand-new topic (and so a brand-new, empty
subscriber set) every time start_topic() ran would silently orphan every
already-connected subscriber the moment a second message arrived — the
first message might display fine (a subscriber connecting *after*
generation had already started still attaches correctly), but every
one after it would show nothing at all to a member who'd been sitting
in the chat the whole time. That was a real, shipped bug: a subscriber
who connects *before* generation has ever started for a conversation
must still correctly receive whatever comes next.

Deliberately just a module-level dict, not a class — there is exactly
one hub for the whole process, matching how title_service's
_background_title_tasks set is also just a bare module-level structure
rather than a singleton class instance.

Documented limitation: this is purely in-process memory, with no
Redis/broker anywhere in this repo. A subscriber whose request lands on
a different app instance than the one running the generation task (see
app.services.instance_pool — relevant once an admin raises
instance_count above its default of 1) receives nothing from this hub.
"Real" channel delivery mode (app.services.chat_settings_service.get_channel_delivery_mode)
is therefore only reliable with a single instance; "cheap" (DB-polling)
mode has no such limitation, since every instance reads the same
database. This is a stated limitation of a first version, not something
silently worked around by inventing new infra.
"""

import asyncio
from dataclasses import dataclass

# Bounded so one badly-lagging subscriber can't grow without limit — a
# full queue just drops that one event (see publish below) rather than
# blocking every other subscriber or the generation task itself. Rare in
# practice, and harmless when it happens: the database stays the
# authoritative record, and both the poller and the /subscribe client do
# a final reconcile fetch once a stream ends (see chat.js).
_QUEUE_MAXSIZE = 1000


@dataclass
class GenerationState:
    """In-memory-only snapshot of a conversation's current (or most
    recent) generation, for a subscriber that connects mid-stream to
    sync against immediately instead of showing a blank bubble until the
    next chunk arrives. The database (Message.content/status) remains
    the durable source of truth — this is purely a convenience."""

    message_id: str
    content: str = ""
    status: str = "streaming"
    sources: list[str] | None = None


class _Topic:
    __slots__ = ("state", "subscribers")

    def __init__(self) -> None:
        # None until the conversation's first-ever generation calls
        # start_topic() — a subscriber connecting before that (see
        # this module's own docstring) still gets a real, live-updating
        # queue, it just has nothing to sync against yet.
        self.state: GenerationState | None = None
        self.subscribers: set[asyncio.Queue] = set()


# conversation_id -> _Topic, created on first subscribe() or
# start_topic() and never removed afterward — cheap to keep around
# (a handful of fields plus whatever subscribers are actually
# connected), and removing it would risk deleting it out from under a
# still-connected subscriber. See this module's own docstring for why
# reusing the same _Topic object across every message in a conversation
# (rather than replacing it) is the actual fix, not just a convenience.
_topics: dict[str, _Topic] = {}


def _get_or_create_topic(conversation_id: str) -> _Topic:
    topic = _topics.get(conversation_id)
    if topic is None:
        topic = _Topic()
        _topics[conversation_id] = topic
    return topic


def start_topic(conversation_id: str, message_id: str) -> None:
    """Begins a new generation for `conversation_id` — resets its
    snapshot to a fresh, empty one for `message_id`, but (unlike an
    earlier version of this function) never touches `subscribers`: every
    connection already watching this conversation keeps working exactly
    as before, now receiving events for the new message instead of the
    previous one."""
    _get_or_create_topic(conversation_id).state = GenerationState(message_id=message_id)


def _publish(conversation_id: str, event: dict) -> None:
    topic = _topics.get(conversation_id)
    if topic is None:
        return
    for queue in topic.subscribers:
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            # See _QUEUE_MAXSIZE above — a dropped event for one lagging
            # subscriber must never block delivery to the others.
            pass


def publish_chunk(conversation_id: str, chunk: str) -> None:
    topic = _topics.get(conversation_id)
    if topic is None or topic.state is None:
        return
    topic.state.content += chunk
    # "message_id" on every event, not just the ones below that used to
    # bother — see this function's own history: a /subscribe connection
    # (any channel member other than the sender) relays this hub's raw
    # events completely unmodified, so if this dict didn't carry its own
    # message_id, that connection's handleReplyLiveEvent (see chat.js)
    # had nothing to key a bubble off of and silently dropped every
    # chunk — a real, shipped bug, not a hypothetical one. Only
    # stream_reply's own relay (the sender's tab) additionally stamps
    # this same key on top, which is harmless (same value either way).
    _publish(conversation_id, {"chunk": chunk, "message_id": topic.state.message_id})


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
    """Tells every live watcher (the sender's own relay in
    reply_generation_service.stream_reply, and any other member's
    /subscribe connection) that this conversation's in-flight reply was
    deleted out from under them — see
    app.services.conversation_service.delete_message, the only caller.
    Mirrors publish_error/publish_done in every way that matters,
    "message_id" included: it updates the snapshot a late-joining
    subscriber's sync reads, and its "deleted" key is a stopping
    condition for both relay loops, exactly like "done"/"error"."""
    topic = _topics.get(conversation_id)
    if topic is None or topic.state is None:
        return
    topic.state.status = "deleted"
    _publish(conversation_id, {"deleted": True, "message_id": topic.state.message_id})


def subscribe(conversation_id: str) -> tuple[GenerationState | None, asyncio.Queue]:
    """Registers a new subscriber for `conversation_id` — creating its
    topic if this is the very first subscriber or publisher it's ever
    had — and returns the topic's current snapshot (None if no
    generation has started for it yet) alongside the queue future events
    will arrive on. Correct to call before any generation has ever
    happened for this conversation: the returned queue is still fully
    live and will receive whatever the *next* start_topic()/publish_*
    call sends, exactly like a subscriber that connects mid-stream (see
    this module's own docstring for why this matters — it's the actual
    fix for a real bug, not a hypothetical). The snapshot is a shallow
    reference, not a copy — safe to read once here since the caller only
    inspects it immediately after subscribing, before any further
    mutation could occur (this is all single-threaded asyncio)."""
    topic = _get_or_create_topic(conversation_id)
    queue: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
    topic.subscribers.add(queue)
    return topic.state, queue


def unsubscribe(conversation_id: str, queue: asyncio.Queue) -> None:
    topic = _topics.get(conversation_id)
    if topic is not None:
        topic.subscribers.discard(queue)
