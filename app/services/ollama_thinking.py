"""
Detects and strips a "thinking"-capable model's raw reasoning out of its
reply — split out of app/services/ollama_client.py (which calls both
functions here from chat_stream) purely to keep that file under
CLAUDE.md's file-size rule; this is otherwise a fully self-contained
concern with no dependency on the rest of that file.
"""

from collections.abc import AsyncGenerator

import httpx

from app.services import ollama_pool

_REQUEST_TIMEOUT = httpx.Timeout(120.0, connect=5.0)

# Cache of model tag -> whether Ollama reports it as thinking-capable
# (see model_supports_thinking below) — the answer is a property of the
# model itself and never changes for a given tag within the process's
# lifetime, so a query result is reused for every later call rather than
# re-fetching /api/show on every single chat request.
_thinking_capability_cache: dict[str, bool] = {}


async def model_supports_thinking(model: str) -> bool:
    """Whether Ollama reports `model` as capable of "thinking" (extended
    reasoning before its final answer) — see ollama_client.chat_stream,
    which only applies strip_inline_thinking's buffering for a model
    this returns True for, so a normal model's reply keeps streaming
    incrementally with zero added latency. Fails safe (False, i.e. "skip
    the filter, stream as usual") on any error querying Ollama — a
    reasoning-only model streaming its raw thinking text unfiltered on a
    momentary /api/show hiccup is a far smaller problem than either
    crashing the chat entirely or silently buffering an ordinary model's
    whole reply. Queried against a single pool-picked host (see
    app.services.ollama_pool) rather than every host — a model's
    capabilities don't vary by which Ollama instance answers, so there's
    nothing to gain from spreading this particular call out; picking one
    (rather than a fixed host) just keeps this correct once "remote" mode
    means there's no single fixed host to rely on anymore."""
    if model in _thinking_capability_cache:
        return _thinking_capability_cache[model]
    host = ollama_pool.pick_host()
    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
        try:
            resp = await client.post(f"{host}/api/show", json={"model": model})
            resp.raise_for_status()
            capabilities = resp.json().get("capabilities", [])
        except httpx.HTTPError:
            return False
    supports = "thinking" in capabilities
    _thinking_capability_cache[model] = supports
    return supports


# Some "thinking"-capable models (see app/model_catalog.py's DictaLM
# entries) don't separate reasoning into Ollama's own `message.thinking`
# field the way some other thinking models do — confirmed directly
# against Ollama's API for dicta-il/dictalm-3.0-1.7b-thinking: the
# *entire* response, reasoning included, streams through
# `message.content` instead, with a literal "</think>" marker ending the
# reasoning portion partway through it (no matching "<think>" ever
# appears in the stream — the model's own chat template injects that
# invisibly as a forced prefix before generation even starts). Without
# stripping this out, a reply would show the model thinking out loud
# before ever reaching its real answer.
_THINK_CLOSE_TAG = "</think>"


async def strip_inline_thinking(chunks: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
    """Wraps a raw content-chunk stream, discarding everything up to and
    including a "</think>" marker if the model emits one inline (see
    _THINK_CLOSE_TAG above), then passing every chunk after that through
    unchanged. Deliberately only ever called for a model
    model_supports_thinking has confirmed is thinking-capable (see
    ollama_client.chat_stream) — nothing here can tell "still reasoning,
    a marker might still show up" apart from "no marker ever coming"
    from chunk content alone, so it has no choice but to buffer
    *everything* until it knows which, which would silently kill live
    incremental streaming for every other model if applied
    unconditionally to all of them. If the marker never actually shows
    up (a corrupted/unexpected response from a model this was still
    applied to), whatever was buffered is flushed once the stream ends
    rather than silently dropped.
    """
    pending = ""
    async for chunk in chunks:
        pending += chunk
        close_idx = pending.find(_THINK_CLOSE_TAG)
        if close_idx != -1:
            after = pending[close_idx + len(_THINK_CLOSE_TAG) :]
            if after:
                yield after
            async for later_chunk in chunks:
                yield later_chunk
            return
    if pending:
        yield pending
