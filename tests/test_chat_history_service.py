"""Unit tests for app/services/chat_history_service.py — the pure history-trimming/merging helpers split out
of chat_service.py under CLAUDE.md's file-size rule (see that module's own docstring)."""

from app.models import Message
from app.services.chat_history_service import trim_history


def _msg(role: str, content: str, sender_id: str | None = None) -> Message:
    return Message(role=role, content=content, sender_id=sender_id)


def test_trim_history_keeps_everything_when_it_fits():
    messages = [_msg("user", "hi"), _msg("assistant", "hello")]
    trimmed = trim_history(messages, num_ctx=4096)
    assert [m["content"] for m in trimmed] == ["hi", "hello"]


def test_trim_history_drops_oldest_first_when_over_budget():
    # num_ctx=256 (the function's own floor) minus 512 reserved clamps to
    # a 256-token floor -> 1024 characters of budget (256 * 4 chars/token).
    old = _msg("user", "x" * 900)
    newer = _msg("assistant", "y" * 900)
    newest = _msg("user", "z" * 100)
    trimmed = trim_history([old, newer, newest], num_ctx=256)

    contents = [m["content"] for m in trimmed]
    assert old.content not in contents  # dropped: budget exceeded once it's included
    assert newest.content in contents  # newest is always kept, even alone
    assert contents.index(newer.content) < contents.index(newest.content)  # order preserved


def test_trim_history_always_keeps_at_least_the_newest_message():
    """Even a single message longer than the whole budget must not be
    dropped — an empty history would mean the model sees no prompt at
    all, which is worse than one slightly-over-budget message."""
    huge = _msg("user", "x" * 100_000)
    trimmed = trim_history([huge], num_ctx=256)
    assert len(trimmed) == 1


def test_trim_history_leaves_single_message_turns_untouched():
    """The overwhelmingly common case (a personal chat, or a channel message that was itself the one that
    asked the AI) must come out byte-for-byte identical to before consecutive-role merging existed — no
    sender prefix, no joining, even though every "user" message here does have a sender_id set."""
    messages = [_msg("user", "hi", sender_id="u1"), _msg("assistant", "hello")]
    trimmed = trim_history(messages, num_ctx=4096)
    assert trimmed == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]


def test_trim_history_merges_consecutive_user_messages_with_sender_prefixes():
    """The real bug this covers: a channel's shared conversation can accumulate several "user"-role messages
    in a row (any member can post with ask_ai=False — see chat_service.build_reply_stream's own docstring),
    which a real, strict chat template (Mistral/Ministral's) rejects outright — confirmed live against
    Ollama's own server logs: "After the optional system message, conversation roles must alternate user and
    assistant roles". Sender display names can't come from a bare in-memory Message here (sender_display_name
    reads message.sender.username, which needs a real DB-loaded relationship) — covered instead by
    test_chat_service.py's build_reply_stream-level tests, which use real committed Users. This test only
    proves the merge/grouping itself: consecutive same-role runs collapse into one turn, non-consecutive ones
    don't."""
    messages = [
        _msg("user", "first"),
        _msg("user", "second"),
        _msg("assistant", "reply"),
        _msg("user", "third"),
    ]
    trimmed = trim_history(messages, num_ctx=4096)
    assert [m["role"] for m in trimmed] == ["user", "assistant", "user"]
    assert trimmed[0]["content"] == "first\nsecond"
    assert trimmed[2]["content"] == "third"
