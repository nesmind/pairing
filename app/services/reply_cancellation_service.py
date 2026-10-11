"""Cancellation bookkeeping for in-progress replies, split from reply_generation_service for the file-size
rule: the two pieces of state conversation_service.delete_message needs from outside.

Two registries:
- `_active_generation_tasks`: message_id -> its asyncio.Task, so cancel_generation can cancel from outside
  (an admin deleting someone's in-progress reply; stream_reply's cancel_on_disconnect covers the sender disconnecting).
- `_deleted_message_ids`: ids conversation_service.delete_message took down, checked by every write in
  reply_generation_service (_mark_error, _run_generation's success path) before persisting. A "read the DB,
  see not deleted, write" guard is racy: two commits (delete_message's and this module's write for the cancelled
  reply) can interleave either way. mark_message_deleted must run *before* cancel_generation, synchronously with
  no `await` between; see its docstring.
"""

import asyncio

_active_generation_tasks: dict[str, asyncio.Task] = {}
_deleted_message_ids: set[str] = set()


def register_task(message_id: str, task: asyncio.Task) -> None:
    """Called right after a generation task is created (reply_generation_service._schedule_generation); removed by
    the task's own done-callback.
    """
    _active_generation_tasks[message_id] = task
    task.add_done_callback(lambda _t, mid=message_id: _active_generation_tasks.pop(mid, None))


def cancel_generation(message_id: str) -> bool:
    """Cancels an in-progress reply's task, used by conversation_service.delete_message *after*
    mark_message_deleted. Unlike cancel_on_disconnect, works whoever is connected. False if nothing to cancel.
    """
    task = _active_generation_tasks.get(message_id)
    if task is None or task.done():
        return False
    task.cancel()
    return True


def mark_message_deleted(message_id: str) -> None:
    """Call *before* cancel_generation, synchronously (no `await` between); the ordering is load-bearing (see module
    docstring)."""
    _deleted_message_ids.add(message_id)


def is_message_deleted(message_id: str) -> bool:
    return message_id in _deleted_message_ids
