"""Detects and strips a "thinking" model's raw reasoning from its reply. Split from ollama_client.py (which calls
both functions from chat_stream) for the file-size rule; self-contained.
"""

from collections.abc import AsyncGenerator

import httpx

from app.services import ollama_pool

_REQUEST_TIMEOUT = httpx.Timeout(120.0, connect=5.0)

# model tag -> whether Ollama reports it thinking-capable (model_supports_thinking). A property of the model,
# fixed for the process lifetime, so reused instead of re-fetching /api/show on every chat request.
_thinking_capability_cache: dict[str, bool] = {}


async def model_supports_thinking(model: str) -> bool:
    """Whether Ollama reports `model` as "thinking"-capable (extended reasoning before the answer).
    ollama_client.chat_stream applies strip_inline_thinking's buffering only for models returning True, so normal
    models keep streaming incrementally with no added latency. Fails safe (False: stream as usual) on any
    error: a reasoning model streaming raw thinking on a momentary /api/show hiccup is much smaller than crashing
    the chat or buffering an ordinary model's whole reply. Asks one pool-picked host (ollama_pool): capabilities
    don't vary by instance, and picking one (not a fixed host) stays correct in "remote" mode.
    """
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


# Some "thinking" models (app/model_catalog.py's DictaLM entries) don't use Ollama's `message.thinking`:
# checked against dicta-il/dictalm-3.0-1.7b-thinking, the *entire* response, reasoning included, streams in
# `message.content`, with a literal "</think>" ending the reasoning (no "<think>" appears: the chat template
# injects it as a forced prefix). Unstripped, a reply would show the model thinking out loud first.
_THINK_CLOSE_TAG = "</think>"


async def strip_inline_thinking(chunks: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
    """Wraps a raw content-chunk stream, discarding everything up to and including an inline "</think>"
    (_THINK_CLOSE_TAG), then passing the rest through. Called only for models model_supports_thinking confirmed
    (ollama_client.chat_stream): chunk content can't tell "still reasoning" from "no marker coming", so it must
    buffer everything until it knows, which would kill incremental streaming if applied to all models. If the
    marker never appears, the buffer is flushed at stream end, not dropped.
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
