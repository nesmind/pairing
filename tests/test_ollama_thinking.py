"""Tests for services/ollama_thinking.py's strip_inline_thinking, the filter discarding a "thinking" model's raw
reasoning when it leaks into `message.content` instead of Ollama's `message.thinking` (seen with
dicta-il/dictalm-3.0-1.7b-thinking; see _THINK_CLOSE_TAG in the module).
"""

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
    # The marker arrives split across two chunks: the scenario the trailing-lag buffer exists for.
    result = await _collect(["thinking out loud </thi", "nk>", "the real answer"])
    assert result == "the real answer"


@pytest.mark.asyncio
async def test_passes_through_unchanged_when_no_close_tag_ever_appears():
    """The common case: a normal (or properly separated-thinking) model whose content never contains "</think>":
    nothing is dropped, just possibly reassembled across chunks.
    """
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
