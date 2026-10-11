"""Tests for services/title_service.py: "simple" mode's pure summarization helper, "smart" mode's model-based
generator with its truncated-reasoning guard, and the fire-and-forget scheduling that keeps it running after its request
ends.
"""

import asyncio

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.database import Base
from app.models import Conversation
from app.services import title_service
from app.services.title_service import (
    _summarize_as_title,
    maybe_generate_title_with_model,
    maybe_set_title,
    schedule_smart_title_generation,
)


def test_summarize_as_title_keeps_a_short_message_whole():
    assert _summarize_as_title("Hi there") == "Hi there"


def test_summarize_as_title_collapses_whitespace():
    assert _summarize_as_title("Hello\n\nworld   there") == "Hello world there"


def test_summarize_as_title_truncates_long_text_without_a_sentence_break():
    text = "a" * 58 + " " + "b" * 10
    assert _summarize_as_title(text) == "a" * 58 + "…"


def test_summarize_as_title_prefers_the_first_sentence_when_short_enough():
    text = "Short question here. " + "b" * 100
    assert _summarize_as_title(text) == "Short question here."


def test_summarize_as_title_strips_a_code_fence_entirely():
    assert _summarize_as_title("```python\ndef foo():\n    pass\n```") == "New chat"


def test_summarize_as_title_strips_code_fence_but_keeps_surrounding_text():
    text = "Please review this:\n```python\ndef foo():\n    pass\n```\nThanks!"
    assert _summarize_as_title(text) == "Please review this: Thanks!"


def test_summarize_as_title_strips_inline_code_backticks_but_keeps_content():
    assert _summarize_as_title("How do I use `printf` in C?") == "How do I use printf in C?"


def test_summarize_as_title_falls_back_to_new_chat_for_empty_input():
    assert _summarize_as_title("   \n\n  ") == "New chat"


def test_summarize_as_title_preserves_non_english_text_as_is():
    """No model call, so no per-language handling: whatever language the message is in carries through, unlike "smart"
    mode's model-prompted approach."""
    text = "מה בירת צרפת?"
    assert _summarize_as_title(text) == text


def test_maybe_set_title_sets_the_title_from_new_chat():
    conversation = Conversation(title="New chat")
    maybe_set_title(conversation, "What is the capital of France?")
    assert conversation.title == "What is the capital of France?"


def test_maybe_set_title_does_not_overwrite_an_existing_title():
    conversation = Conversation(title="Already Named")
    maybe_set_title(conversation, "Some other message")
    assert conversation.title == "Already Named"


@pytest.mark.asyncio
async def test_maybe_generate_title_with_model_accepts_a_short_title(db, monkeypatch):
    async def fake_chat_once(*_args, **_kwargs):
        return "Capital of France"

    monkeypatch.setattr(title_service, "chat_once", fake_chat_once)

    conversation = Conversation(title="New chat")
    await maybe_generate_title_with_model(db, conversation, "What is the capital of France?")
    assert conversation.title == "Capital of France"


@pytest.mark.asyncio
async def test_maybe_generate_title_with_model_rejects_truncated_reasoning_as_a_title(db, monkeypatch):
    """Regression: a "thinking" model whose reasoning doesn't finish within the title token budget never reaches its
    answer, and chat_once returns a fragment of raw reasoning. Saving it would show part of the model's internal
    monologue in the sidebar instead of a real title or the honest "New chat".
    """

    async def fake_chat_once(*_args, **_kwargs):
        return (
            "The user wants a short chat title summarizing the question "
            "about the capital of France, so I should think about what "
            "a good concise title would look like before answering"
        )

    monkeypatch.setattr(title_service, "chat_once", fake_chat_once)

    conversation = Conversation(title="New chat")
    await maybe_generate_title_with_model(db, conversation, "What is the capital of France?")
    assert conversation.title == "New chat"


@pytest.mark.asyncio
async def test_maybe_generate_title_with_model_does_not_overwrite_an_existing_title(db, monkeypatch):
    async def fake_chat_once(*_args, **_kwargs):
        return "Some New Title"

    monkeypatch.setattr(title_service, "chat_once", fake_chat_once)

    conversation = Conversation(title="Already Named")
    await maybe_generate_title_with_model(db, conversation, "Some message")
    assert conversation.title == "Already Named"


@pytest.mark.asyncio
async def test_schedule_smart_title_generation_survives_after_the_caller_moves_on(monkeypatch):
    """Regression: title generation ran as a plain `await` in chat_service.build_reply_stream *after* "done", but
    chat.js stops reading right after "done" and ASGI then cancels the generator, killing that `await` before the model
    call finished. schedule_smart_title_generation must run independently of its caller, landing the title even after
    the caller's scope (simulated as an already-closed `with`-like session) is gone.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def fake_chat_once(*_args, **_kwargs):
        return "Capital of France"

    monkeypatch.setattr(title_service, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr(title_service, "chat_once", fake_chat_once)

    async with session_factory() as setup_db:
        conversation = Conversation(owner_id=None, title="New chat", model="fake-model", params={})
        setup_db.add(conversation)
        await setup_db.commit()
        conversation_id = conversation.id
    # The session that created the conversation is closed: the scheduled task must open its own, like a real request's
    # session gone by then.

    schedule_smart_title_generation(conversation_id, "What is the capital of France?")
    assert title_service._background_title_tasks, "expected a task to have been scheduled"
    await asyncio.gather(*title_service._background_title_tasks)

    async with session_factory() as verify_db:
        updated = await verify_db.get(Conversation, conversation_id)
        assert updated.title == "Capital of France"

    await engine.dispose()


@pytest.mark.asyncio
async def test_schedule_smart_title_generation_does_not_block_the_caller(monkeypatch):
    """schedule_smart_title_generation must return at once, before the model call it starts has begun: the point of a
    background task over a plain `await`."""
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_chat_once(*_args, **_kwargs):
        started.set()
        await release.wait()
        return "Capital of France"

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    monkeypatch.setattr(title_service, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr(title_service, "chat_once", slow_chat_once)

    async with session_factory() as setup_db:
        conversation = Conversation(owner_id=None, title="New chat", model="fake-model", params={})
        setup_db.add(conversation)
        await setup_db.commit()
        conversation_id = conversation.id

    schedule_smart_title_generation(conversation_id, "hello")
    assert not started.is_set()

    release.set()
    await asyncio.gather(*title_service._background_title_tasks)
    await engine.dispose()
