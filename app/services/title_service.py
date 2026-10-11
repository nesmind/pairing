"""Gives a new conversation a real sidebar title instead of "New chat": "simple" (instant local text processing,
maybe_set_title) or "smart" (asks the model, maybe_generate_title_with_model), per the admin mode
(chat_settings_service.get_title_mode). Split from chat_service.py (build_reply_stream calls whichever fits)
for the file-size rule; the two modes needn't be assembled together at call sites.
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

# A fenced code block can span many lines (re.DOTALL) and never reaches a one-line title; an inline code span
# keeps its contents but loses the backticks ("`foo`" reads worse than "foo").
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")

# A sentence-ending mark followed by whitespace or end of string: lets _summarize_as_title cut a long message at a
# sentence boundary, not an arbitrary offset.
_SENTENCE_END_RE = re.compile(r"[.!?…](?:\s|$)")


def _summarize_as_title(message: str) -> str:
    """ "Simple" mode's title: local string processing, no model call. Strips markdown code, collapses whitespace
    (a pasted multi-line message becomes one line), keeps a short message whole, else its first sentence if it
    fits the budget (reads as a title, not a cutoff), else truncates at the last whole word with a trailing "…".
    Being extractive, it needs no per-language handling: the title is in the message's language.
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
    """ "Simple" title mode (chat_settings_service.get_title_mode): gives a conversation's first message a title
    instead of "New chat" (_summarize_as_title). Synchronous, no I/O or commit of its own
    (chat_service.build_reply_stream
    commits with the rest of the turn). Safe on every message: the guard below limits it to once and, as it can't
    fail like a model call, a never-titled conversation isn't stuck that way.
    """
    if conversation.title != "New chat":
        return
    conversation.title = _summarize_as_title(first_user_message)


_HEBREW_LETTER_RE = re.compile(r"[֐-׿]")
_LATIN_LETTER_RE = re.compile(r"[A-Za-z]")

# "Smart" mode's budget for the model's title call. A "thinking" model reasons before its answer, and
# num_predict counts reasoning tokens: a small model can spend its *whole* budget (on slow hardware a minute
# or more) on a one-line title and never write one. chat_stream (under chat_once) already strips raw reasoning
# for models the engine reports thinking-capable (ollama_client's _strip_inline_thinking; needed for
# dicta-il/dictalm-3.0-1.7b-thinking), but that doesn't help if reasoning doesn't *finish* in budget: only a
# truncated fragment remains, which _TITLE_MAX_WORDS catches. Generous to make that rare: it runs *after*
# "done" is yielded (chat_service.build_reply_stream), off the critical path.
_TITLE_NUM_PREDICT = 1000

# The title prompt asks for "max 6 words" (_build_title_prompt); a real title roughly honors it, and a response
# many times longer means reasoning was cut off by _TITLE_NUM_PREDICT, not a verbose title. Comfortably above 6
# for models that round up, far below truncated reasoning prose.
_TITLE_MAX_WORDS = 12


def _build_title_prompt(message: str) -> str:
    """Builds "smart" mode's "invent a short title" instruction in the language of `message` where we can tell.
    An English "reply in Hebrew" is unreliable (models default to English for short output); writing the
    *instruction itself* in Hebrew keeps the model in Hebrew, which continues in its context's language. Hebrew
    only for now.
    """
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
    """ "Smart" title mode (chat_settings_service.get_title_mode): asks the model for a title; reads more naturally
    than _summarize_as_title but costs a second, slower call, occasionally unreliable (see the sanity check below).
    Best-effort: any failure leaves the current title. Called *after* "done" (chat_service.build_reply_stream) so it
    never delays the reply or blocks Send.
    """
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
        # A title this long is almost certainly truncated reasoning (_TITLE_MAX_WORDS); saving it would show part of the
        # model's internal monologue in the sidebar instead of a title or the honest "New chat".
        if title and len(title.split()) <= _TITLE_MAX_WORDS:
            conversation.title = title[:80]
            await db.commit()
    except InferenceError:
        pass


# Strong refs to in-flight "smart" title tasks: asyncio keeps only a weak reference to a bare create_task
# result, so with nothing else referencing it the task could be garbage-collected mid-run and drop the title.
# Each task removes itself via its done-callback.
_background_title_tasks: set[asyncio.Task] = set()


def schedule_smart_title_generation(conversation_id: str, first_user_message: str) -> None:
    """Starts "smart" title generation detached from the request/response (wraps maybe_generate_title_with_model).
    Necessary: after "done", chat_service.build_reply_stream's generator lives only while the HTTP request
    does, and chat.js deliberately disconnects right after "done", so ASGI cancels the generator (and a plain
    `await maybe_generate_title_with_model(...)`), silently dropping the title every time. Uses its own session:
    the request's `db` is tied to the soon-cancelled scope.
    """
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
        # Broad on purpose: a detached task with no caller to propagate to; an unhandled exception would vanish as an
        # "exception was never retrieved" warning without context instead of appearing in the logs.
        # maybe_generate_title_with_model handles its own InferenceError; this backstops anything else (e.g. a DB
        # error).
        logger.exception("Smart-mode title generation failed for conversation %s", conversation_id)
