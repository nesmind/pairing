"""
Pure history-assembly helpers for app/services/chat_service.py's build_reply_stream — split out under
CLAUDE.md's file-size rule (same reasoning as title_service.py: separable enough, and substantial enough on
its own, to keep out of chat_service.py directly). No DB access of its own; both functions take/return plain
data (Message objects in, {"role", "content"} dicts out).
"""

import itertools

from app.models import Message

# Rough characters-per-token ratio for English text. The ML engine doesn't expose
# a cheap client-side tokenizer, so this estimate is used only to decide
# how much history to keep on our side of the wire — the ML engine still applies
# its own real token limit (num_ctx) server-side as the source of truth.
_CHARS_PER_TOKEN = 4


def trim_history(messages: list[Message], num_ctx: int, reserved_tokens: int = 512) -> list[dict]:
    """Keeps only as much recent history as should comfortably fit in the
    model's context window, so a long-running chat doesn't silently lose
    coherence by overflowing num_ctx. `reserved_tokens` leaves headroom
    for the system prompt, RAG context, and the model's own reply.

    Walks from the newest message backwards, keeping messages until the
    running character budget would be exceeded, then reverses back to
    chronological order.
    """
    budget_chars = max(num_ctx - reserved_tokens, 256) * _CHARS_PER_TOKEN
    kept: list[Message] = []
    used = 0
    for message in reversed(messages):
        used += len(message.content)
        if used > budget_chars and kept:
            break
        kept.append(message)
    kept.reverse()
    return _merge_consecutive_same_role(kept)


def _merge_consecutive_same_role(messages: list[Message]) -> list[dict]:
    """Folds any run of consecutive same-role messages into one turn — a channel's shared conversation can
    accumulate several "user"-role messages in a row (any member can post with ask_ai=False; the AI only ever
    replies when a message explicitly asks it to, see build_reply_stream's own docstring), which a real,
    strict chat template (Mistral/Ministral's — used by both Ollama and Matricxon for every model this app
    currently ships) rejects outright: "After the optional system message, conversation roles must alternate
    user and assistant roles" — confirmed live against Ollama's own server logs, not assumed from docs. A
    personal chat never needs this (its owner always asks the AI every time, so history already alternates
    naturally) but paying the cost unconditionally here is simpler than a channel-only branch, and correct
    either way. Single-message turns (the overwhelming common case) are left byte-for-byte identical to
    before this existed; only an actual run of 2+ gets its lines joined, each prefixed with its own sender's
    display name (see Message.sender_display_name) so folding several members' messages into one turn doesn't
    lose who said what — an "assistant"/"system" message has no sender to prefix, so those stay plain."""
    merged: list[dict] = []
    for role, group in itertools.groupby(messages, key=lambda m: m.role):
        group = list(group)
        if len(group) == 1:
            merged.append({"role": role, "content": group[0].content})
            continue
        lines = [f"{m.sender_display_name}: {m.content}" if m.sender_display_name else m.content for m in group]
        merged.append({"role": role, "content": "\n".join(lines)})
    return merged
