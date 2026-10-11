"""Tests for services/chat_history_service.py: the pure history-trimming/merging helpers split from chat_service.py
(file-size rule).
"""

from app.models import Message
from app.services.chat_history_service import trim_history


def _msg(role: str, content: str, sender_id: str | None = None) -> Message:
    return Message(role=role, content=content, sender_id=sender_id)


def test_trim_history_keeps_everything_when_it_fits():
    messages = [_msg("user", "hi"), _msg("assistant", "hello")]
    trimmed = trim_history(messages, num_ctx=4096)
    assert [m["content"] for m in trimmed] == ["hi", "hello"]


def test_trim_history_drops_oldest_first_when_over_budget():
    # num_ctx=256 (the function's floor) minus 512 reserved clamps to a 256-token floor -> 1024 characters (256 * 4
    # chars/token).
    old = _msg("user", "x" * 900)
    newer = _msg("assistant", "y" * 900)
    newest = _msg("user", "z" * 100)
    trimmed = trim_history([old, newer, newest], num_ctx=256)

    contents = [m["content"] for m in trimmed]
    assert old.content not in contents
    assert newest.content in contents
    assert contents.index(newer.content) < contents.index(newest.content)


def test_trim_history_always_keeps_at_least_the_newest_message():
    """Even a single message longer than the whole budget is kept: an empty history means no prompt at all, worse
    than one slightly over budget.
    """
    huge = _msg("user", "x" * 100_000)
    trimmed = trim_history([huge], num_ctx=256)
    assert len(trimmed) == 1


def test_trim_history_leaves_single_message_turns_untouched():
    """The common case (a personal chat, or a channel message that asked the AI) must come out byte-for-byte as before
    role merging existed: no sender prefix or joining, even with sender_id set on every "user" message.
    """
    messages = [_msg("user", "hi", sender_id="u1"), _msg("assistant", "hello")]
    trimmed = trim_history(messages, num_ctx=4096)
    assert trimmed == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]


def test_trim_history_merges_consecutive_user_messages_with_sender_prefixes():
    """Covers a channel conversation accumulating several "user" messages in a row (ask_ai=False, see
    chat_service.build_reply_stream), which a strict chat template (Mistral/Ministral) rejects: "After the
    optional system message, conversation roles must alternate user and assistant roles" (seen in Ollama's logs).
    Sender display names need a DB-loaded relationship, so test_chat_service.py covers them with real Users; this
    proves only the grouping: consecutive same-role runs collapse into one turn, non-consecutive ones don't.
    """
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


def test_failed_replies_are_left_out_but_their_question_stays():
    from app.services.chat_history_service import without_failed_replies

    question = _msg("user", "q1")
    failed = Message(role="assistant", content="part", status="error", error_message="Cancelled")
    ok = Message(role="assistant", content="fine", status="done")

    kept = without_failed_replies([question, failed, _msg("user", "q2"), ok])

    assert [m.content for m in kept] == ["q1", "q2", "fine"]


def test_trim_history_keeps_the_same_first_message_while_the_chat_grows():
    """A sliding window moves the prompt start every turn, so the model's prompt cache never matches. Stepwise
    trimming keeps the start until the next step.
    """
    budget_chars = (4096 - 512) * 4
    history: list[Message] = []
    firsts = []
    for turn in range(80):
        history += [_msg("user", f"q{turn:02d}" + "x" * 297), _msg("assistant", f"a{turn:02d}" + "y" * 297)]
        trimmed = trim_history(history, num_ctx=4096)
        assert sum(len(m["content"]) for m in trimmed) <= budget_chars
        firsts.append(trimmed[0]["content"][:3])
    changes = sum(1 for a, b in zip(firsts, firsts[1:], strict=False) if a != b)
    assert changes <= 14
    assert trimmed[-1]["content"].startswith("a79")


def test_trim_history_never_returns_an_empty_history():
    huge_old = _msg("user", "x" * 5000)
    huge_new = _msg("user", "y" * 5000)
    assert [m["content"][0] for m in trim_history([huge_old, huge_new], num_ctx=256)] == ["y"]
