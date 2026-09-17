"""Unit tests for app/services/ollama_thinking.py's strip_inline_thinking
— the filter that discards a "thinking"-capable model's raw reasoning
when it leaks into `message.content` instead of Ollama's own
`message.thinking` field (confirmed happening for real against
dicta-il/dictalm-3.0-1.7b-thinking). See that module's own
_THINK_CLOSE_TAG docstring for the full story."""

import pytest

from app.services.ollama_thinking import strip_inline_thinking


async def _achunks(chunks: list[str]):
    for chunk in chunks:
        yield chunk


async def _collect(chunks: list[str]) -> str:
    return "".join([c async for c in strip_inline_thinking(_achunks(chunks))])


@pytest.mark.asyncio
async def test_strips_everything_up_to_and_including_the_close_tag():
    result = await _collect(["The user wants a title. ", "</think>", "Capital of France"])
    assert result == "Capital of France"


@pytest.mark.asyncio
async def test_close_tag_split_across_chunk_boundaries_is_still_caught():
    # The marker itself arrives split across two separate chunks — the
    # exact scenario the trailing-lag buffer exists for.
    result = await _collect(["thinking out loud </thi", "nk>", "the real answer"])
    assert result == "the real answer"


@pytest.mark.asyncio
async def test_passes_through_unchanged_when_no_close_tag_ever_appears():
    """The common case: a normal (or a properly-separated-thinking)
    model whose content never contains "</think>" at all — nothing
    should ever be dropped, just possibly reassembled across chunks."""
    chunks = ["Hello", ", ", "world", "!"]
    result = await _collect(chunks)
    assert result == "Hello, world!"


@pytest.mark.asyncio
async def test_content_after_close_tag_streams_normally_in_multiple_chunks():
    result = await _collect(["reasoning", "</think>", "answer ", "part ", "two"])
    assert result == "answer part two"


@pytest.mark.asyncio
async def test_empty_stream_yields_nothing():
    assert await _collect([]) == ""
