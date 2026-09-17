"""Cancellation bookkeeping for in-progress reply generation — split out
of app.services.reply_generation_service purely to keep that file under
CLAUDE.md's file-size rule; this is still that module's own mechanism,
just the two pieces of state conversation_service.delete_message needs
to reach in from outside.

Two independent registries:
- `_active_generation_tasks`: message_id -> its live asyncio.Task, so
  cancel_generation can find and cancel one from outside
  reply_generation_service (an admin deleting someone else's in-progress
  reply, not the sender's own request disconnecting — see
  reply_generation_service.stream_reply's own cancel_on_disconnect for
  that other case).
- `_deleted_message_ids`: message_ids conversation_service.delete_message
  has taken down, checked by every write path in
  reply_generation_service (_mark_error, and _run_generation's own
  success path) before it persists anything. Needed because a plain
  "read the DB, see it's not deleted yet, then write" guard is racy: two
  independent commits (delete_message's own, and this module's write for
  the very reply it just cancelled) can interleave in either order, so
  seeing "not deleted yet" proves nothing about which write actually
  lands last. mark_message_deleted must be called *before*
  cancel_generation, synchronously, with no `await` in between — see its
  own docstring for why that ordering is what actually closes the race.
"""

import asyncio

_active_generation_tasks: dict[str, asyncio.Task] = {}
_deleted_message_ids: set[str] = set()


def register_task(message_id: str, task: asyncio.Task) -> None:
    """Called once, right after a generation task is created (see
    reply_generation_service._schedule_generation) — removed by the
    task's own done-callback, so this never outlives it."""
    _active_generation_tasks[message_id] = task
    task.add_done_callback(lambda _t, mid=message_id: _active_generation_tasks.pop(mid, None))


def cancel_generation(message_id: str) -> bool:
    """Cancels an in-progress reply's task outright — used by
    conversation_service.delete_message, always *after*
    mark_message_deleted below; unlike cancel_on_disconnect, works
    regardless of who's still connected. False if nothing to cancel."""
    task = _active_generation_tasks.get(message_id)
    if task is None or task.done():
        return False
    task.cancel()
    return True


def mark_message_deleted(message_id: str) -> None:
    """Call *before* cancel_generation, synchronously (no `await` in
    between) — see this module's own docstring for why the ordering is
    load-bearing, not just tidy."""
    _deleted_message_ids.add(message_id)


def is_message_deleted(message_id: str) -> bool:
    return message_id in _deleted_message_ids
