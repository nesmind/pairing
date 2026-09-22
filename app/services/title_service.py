"""
Gives a brand-new conversation a real sidebar title instead of the
"New chat" placeholder — either "simple" (instant local text processing,
see maybe_set_title) or "smart" (asks the model itself, see
maybe_generate_title_with_model), per the admin-configured mode (see
app.services.chat_settings_service.get_title_mode). Split out of
app/services/chat_service.py, whose build_reply_stream calls into
whichever of these two fits, at the point that mode needs — this file
existing is just what keeps both under CLAUDE.md's file-size rule, not a
sign these two modes need to be assembled together at every call site.
"""

import asyncio
import logging
import re

from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AsyncSessionLocal
from app.models import Conversation
from app.services.inference_client import InferenceError, chat_once

logger = logging.getLogger("llama_chat")

_TITLE_MAX_LENGTH = 60

# A fenced code block spans potentially many lines (re.DOTALL), so it
# never survives into a one-line title at all; an inline code span
# keeps its contents but loses the backticks, since "`foo`" as a title
# reads worse than plain "foo" for something this short.
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")

# Matches a sentence-ending punctuation mark followed by whitespace or
# end-of-string — used to prefer cutting a long message at a real
# sentence boundary (see _summarize_as_title) rather than an arbitrary
# character offset.
_SENTENCE_END_RE = re.compile(r"[.!?…](?:\s|$)")


def _summarize_as_title(message: str) -> str:
    """ "Simple" mode's title: entirely local string processing, no model
    call. Strips markdown code (unreadable squeezed into one line),
    collapses all whitespace (a multi-line pasted message becomes one
    line), then keeps the message whole if it's already short, otherwise
    keeps just its first sentence if that alone fits the length budget
    (reads as an actual title rather than a mid-thought cutoff), and
    failing that truncates at the last whole word that fits, with a
    trailing "…". Being purely extractive means it needs no per-language
    handling the way "smart" mode's prompt does — whatever language the
    message itself is in is what ends up in the title, automatically.
    """
    text = _CODE_FENCE_RE.sub(" ", message)
    text = _INLINE_CODE_RE.sub(r"\1", text)
    text = " ".join(text.split())
    if not text:
        return "New chat"
    if len(text) <= _TITLE_MAX_LENGTH:
        return text

    sentence_match = _SENTENCE_END_RE.search(text)
    if sentence_match and sentence_match.end() <= _TITLE_MAX_LENGTH:
        return text[: sentence_match.end()].strip()

    truncated = text[:_TITLE_MAX_LENGTH]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated.rstrip(" ,;:-") + "…"


def maybe_set_title(conversation: Conversation, first_user_message: str) -> None:
    """ "Simple" title mode (see
    app.services.chat_settings_service.get_title_mode) — the first time a
    conversation gets a message, gives it a real title instead of the
    "New chat" placeholder (see _summarize_as_title above). Plain and
    synchronous — no I/O of its own, no separate commit: the caller
    (chat_service.build_reply_stream) already commits alongside
    everything else it saves for this turn. Safe to call on every
    message, not just the first: the guard below is what actually limits
    it to once, which — since this can't meaningfully fail the way a
    model call could — also means a conversation that somehow never got
    titled isn't stuck that way forever."""
    if conversation.title != "New chat":
        return
    conversation.title = _summarize_as_title(first_user_message)


_HEBREW_LETTER_RE = re.compile(r"[֐-׿]")
_LATIN_LETTER_RE = re.compile(r"[A-Za-z]")

# "Smart" mode's budget for the model's own title-invention call. A
# "thinking"-capable model reasons before ever writing its real answer,
# and num_predict counts tokens spent reasoning too — a small model can
# burn its *entire* budget (and, on slow hardware, a full minute or
# more) reasoning about a one-line title and never reach any actual
# title text at all. chat_stream (which chat_once below calls into)
# already strips a model's raw reasoning out unconditionally for any
# model Ollama reports as thinking-capable (see
# app/services/ollama_client.py's _strip_inline_thinking — confirmed
# necessary for real against dicta-il/dictalm-3.0-1.7b-thinking), but
# that doesn't help if reasoning simply doesn't *finish* within this
# budget — there's no real title yet to strip down to in that case, only
# a truncated fragment of reasoning, which _TITLE_MAX_WORDS below is
# what actually catches. Generous specifically to make that rare: this
# call runs *after* "done" is yielded (see chat_service.build_reply_stream),
# so it's off the critical path — spending more tokens/time on it
# doesn't block anything user-visible.
_TITLE_NUM_PREDICT = 1000

# A title prompt explicitly asks for "max 6 words" (see
# _build_title_prompt) — a real title honors that loosely; a response
# many times longer is a reliable sign that reasoning got cut off by
# _TITLE_NUM_PREDICT before ever reaching a real answer, not an
# unusually verbose title. Comfortably above 6 to tolerate a model that
# rounds up a little, nowhere near what truncated reasoning prose
# actually looks like.
_TITLE_MAX_WORDS = 12


def _build_title_prompt(message: str) -> str:
    """Builds "smart" mode's "invent a short title" instruction, phrased
    in the same language as `message` where we can tell. This matters
    more than it sounds: asking an English instruction like "reply in
    Hebrew" for the title is unreliable — models tend to default back to
    English for short-form output regardless. Writing the *instruction
    itself* in Hebrew reliably keeps the model in Hebrew, since it just
    continues in whatever language its immediate context is already in.
    Only Hebrew gets this treatment for now (see ROADMAP.md for
    extending the same pattern to other RTL/non-English languages)."""
    hebrew_count = len(_HEBREW_LETTER_RE.findall(message))
    latin_count = len(_LATIN_LETTER_RE.findall(message))
    if hebrew_count > latin_count:
        return (
            "כתוב כותרת קצרה (עד 6 מילים) בעברית לשיחה הבאה. "
            "אל תתרגם לאנגלית. ללא מרכאות וללא סימני פיסוק בסוף.\n\n"
            f"{message}"
        )
    return (
        "Summarize the following message as a short chat title "
        "(max 6 words, no quotes, no punctuation at the end):\n\n"
        f"{message}"
    )


async def maybe_generate_title_with_model(
    db: AsyncSession,
    conversation: Conversation,
    first_user_message: str,
) -> None:
    """ "Smart" title mode (see
    app.services.chat_settings_service.get_title_mode) — asks the model
    itself to invent a title, which can read more naturally than
    _summarize_as_title's plain extraction, at the cost of a second,
    slower generation call that's occasionally still unreliable (see the
    sanity check below). Best-effort: any failure just leaves whatever
    title is already there in place. Called *after* "done" is yielded
    (see chat_service.build_reply_stream) so this never delays the reply
    itself or blocks the Send button for the next message."""
    if conversation.title != "New chat":
        return
    try:
        title = await chat_once(
            conversation.model,
            [{"role": "user", "content": _build_title_prompt(first_user_message)}],
            params={
                "temperature": 0.3,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "num_ctx": 2048,
                "num_predict": _TITLE_NUM_PREDICT,
                "seed": -1,
            },
        )
        title = title.strip().strip('"').strip()
        # A title this long is almost certainly truncated reasoning, not
        # an unusually wordy title (see _TITLE_MAX_WORDS) — saving it
        # anyway would leave the sidebar showing a chunk of the model's
        # internal monologue instead of either a real title or the
        # honest "New chat" placeholder.
        if title and len(title.split()) <= _TITLE_MAX_WORDS:
            conversation.title = title[:80]
            await db.commit()
    except InferenceError:
        pass


# Strong references to in-flight "smart" mode title-generation tasks —
# asyncio only holds a *weak* reference to a bare asyncio.create_task()
# result; with nothing else referencing the returned Task, it can be
# garbage-collected mid-run, silently dropping the title generation it
# was doing. Each task removes itself once it's done (success or
# failure) via its own done-callback below, so this doesn't grow
# forever.
_background_title_tasks: set[asyncio.Task] = set()


def schedule_smart_title_generation(conversation_id: str, first_user_message: str) -> None:
    """Kicks off "smart" mode's title generation fully detached from the
    current request/response (see maybe_generate_title_with_model, which
    this wraps) — necessary, not just a nice-to-have: once "done" is
    yielded, chat_service.build_reply_stream's generator only keeps
    running as long as the underlying HTTP request/response stays alive,
    and chat.js is deliberately written to disconnect right after
    reading "done" — ASGI cancels the generator (and any `await` still
    in flight in it, which a plain `await maybe_generate_title_with_model(...)`
    here would have been) the moment that disconnect is detected,
    silently dropping the title every time a client behaves exactly the
    way it's now meant to. Runs against its own fresh session — the
    request's `db` is tied to that same soon-to-be-cancelled scope, so
    it can't be reused here either."""
    task = asyncio.create_task(_generate_title_in_background(conversation_id, first_user_message))
    _background_title_tasks.add(task)
    task.add_done_callback(_background_title_tasks.discard)


async def _generate_title_in_background(conversation_id: str, first_user_message: str) -> None:
    try:
        async with AsyncSessionLocal() as db:
            conversation = await db.get(Conversation, conversation_id)
            if conversation is not None:
                await maybe_generate_title_with_model(db, conversation, first_user_message)
    except Exception:
        # Broad on purpose: this is a detached background task with no
        # caller left to propagate a failure to — an unhandled exception
        # here would otherwise just vanish as an "exception was never
        # retrieved" warning with no context, rather than something
        # findable in the logs. maybe_generate_title_with_model already
        # handles its own InferenceError; this is the backstop for anything
        # else (a DB error, say).
        logger.exception("Smart-mode title generation failed for conversation %s", conversation_id)
