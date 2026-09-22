"""
Model-catalog administration against Ollama's "primary" host (the first
of app.services.ollama_pool.get_effective_hosts() — whichever host is
actually in effect right now, local or the first configured remote one):
pulling and deleting models, for Settings' admin-only model management.
Split out of app/services/ollama_client.py (which stays focused on
actual inference — chat/embeddings, routed across the full pool) purely
to keep both files under CLAUDE.md's file-size rule.

Deliberately NOT routed across the pool the way chat/embeddings are:
which models are installed isn't guaranteed uniform across multiple
Ollama instances, so a pull/delete here only ever targets one explicit
host — mirroring it to the rest of the pool, if there is one, is a
manual step for now (see ROADMAP.md's "Concurrency: running multiple
Ollama instances").
"""

import json

import httpx

from app.services import ollama_pool
from app.services.ollama_client import OllamaError

# Same values as app/services/ollama_client.py's own — duplicated rather
# than imported, so this file doesn't reach into that one's private
# constants just to share two httpx.Timeout objects.
_REQUEST_TIMEOUT = httpx.Timeout(120.0, connect=5.0)
_PULL_TIMEOUT = httpx.Timeout(None, connect=5.0)


def _primary_host() -> str:
    return ollama_pool.get_effective_hosts()[0]


async def pull_model_stream(tag: str):
    """Downloads a model into Ollama, yielding progress updates as they
    arrive. Each yielded dict mirrors Ollama's own NDJSON pull-progress
    shape (`status`, and usually `completed`/`total` byte counts) — the
    caller (see app/routers/settings.py) forwards these to the browser
    over SSE so a "Pull" button can show a real progress bar instead of
    an indefinite spinner during a multi-gigabyte download.
    """
    async with httpx.AsyncClient(timeout=_PULL_TIMEOUT) as client:
        try:
            async with client.stream(
                "POST",
                f"{_primary_host()}/api/pull",
                json={"model": tag, "stream": True},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        raise OllamaError(data["error"])
                    yield data
        except httpx.HTTPError as exc:
            raise OllamaError(f"Pulling {tag} failed: {exc}") from exc


async def delete_model(tag: str) -> None:
    """Removes a model Ollama already has pulled, freeing its disk space.
    Used by the admin-only "Uninstall" action in Settings (see
    app/routers/settings.py) — the inverse of pull_model_stream. Ollama's
    delete endpoint takes a DELETE with a JSON body rather than the tag
    in the URL, which sidesteps having to URL-encode tags containing
    slashes (e.g. `hf.co/unsloth/MiniMax-M3-GGUF:...`)."""
    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
        try:
            resp = await client.request(
                "DELETE",
                f"{_primary_host()}/api/delete",
                json={"model": tag},
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(f"Deleting {tag} failed: {exc}") from exc
