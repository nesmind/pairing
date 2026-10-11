"""Pure history-assembly helpers for chat_service.build_reply_stream, split out for the file-size rule
(like title_service.py). No DB access: Message objects in, {"role", "content"} dicts out.
"""

import itertools

from app.models import Message

# Rough characters per token for English. The engine offers no cheap client-side tokenizer, so this only
# sizes history locally; the engine still enforces num_ctx.
_CHARS_PER_TOKEN = 4


def only_addressed_to_ai(messages: list[Message]) -> list[Message]:
    """Channel history without messages posted without asking the AI (channel_history_setting): a user message
    stays only if an assistant reply answers it (failed counts). The current question is added by the caller.
    """
    answered = {m.reply_to_message_id for m in messages if m.role == "assistant" and m.reply_to_message_id}
    return [m for m in messages if m.role != "user" or m.id in answered]


def without_failed_replies(messages: list[Message]) -> list[Message]:
    """Drops failed or cancelled replies (status "error"): partial text plus a notice, not worth re-reading, and
    costly each turn. The question stays so the next reply can answer it.
    """
    return [m for m in messages if m.status != "error"]


# Trimming advances in steps of this fraction of the budget (see trim_history).
_TRIM_STEPS = 4


def trim_history(messages: list[Message], num_ctx: int, reserved_tokens: int = 512) -> list[dict]:
    """Keeps only the recent history that fits the context window, so long chats don't overflow num_ctx.
    `reserved_tokens` leaves room for the system prompt, RAG context and the reply.

    Oldest messages go first, in steps: the history starts only where the running character count crosses a
    multiple of a quarter of the budget. Start points depend only on earlier messages, so a growing chat keeps
    the same first message for many turns. The prompt cache matches from the start, and a sliding window made
    it re-read the whole chat every turn.
    """
    budget_chars = max(num_ctx - reserved_tokens, 256) * _CHARS_PER_TOKEN
    step = max(budget_chars // _TRIM_STEPS, 1)
    sizes = [len(m.content) for m in messages]
    total = sum(sizes)
    before = 0
    for index in range(len(messages)):
        starts_step = index == 0 or before // step > (before - sizes[index - 1]) // step
        if starts_step and total - before <= budget_chars:
            return _merge_consecutive_same_role(messages[index:])
        before += sizes[index]
    return _merge_consecutive_same_role(_newest_that_fit(messages, budget_chars))


def _newest_that_fit(messages: list[Message], budget_chars: int) -> list[Message]:
    """The newest messages within the budget; always at least the last one."""
    kept: list[Message] = []
    used = 0
    for message in reversed(messages):
        used += len(message.content)
        if used > budget_chars and kept:
            break
        kept.append(message)
    kept.reverse()
    return kept


def _merge_consecutive_same_role(messages: list[Message]) -> list[dict]:
    """Folds runs of same-role messages into one turn. A channel can hold several "user" messages in a row
    (ask_ai=False; the AI replies only when asked), which strict chat templates (Mistral/Ministral, used by
    Ollama and Matricxon) reject: "conversation roles must alternate" (seen in Ollama's logs). Personal chats
    already alternate; paying the cost unconditionally is simpler. Single messages are unchanged; runs of 2+
    are joined with each line prefixed by the sender's display name (Message.sender_display_name).
    Assistant/system messages stay plain.
    """
    merged: list[dict] = []
    for role, group in itertools.groupby(messages, key=lambda m: m.role):
        group = list(group)
        if len(group) == 1:
            merged.append({"role": role, "content": group[0].content})
            continue
        lines = [f"{m.sender_display_name}: {m.content}" if m.sender_display_name else m.content for m in group]
        merged.append({"role": role, "content": "\n".join(lines)})
    return merged
