"""
Model-catalog administration against Matricxon's "primary" host (mirrors app/services/ollama_admin.py's identical
role for Ollama — see that module's own docstring for the full reasoning, unchanged here): pulling and deleting
models, for Settings' admin-only model management. Matricxon's own /api/pull and /api/delete were built to the
same wire shape as Ollama's (see ../matricxon/app/routers/pull_router.py|delete_router.py and
../matricxon/app/pull/tag.py, which resolves the identical `hf.co/<repo>:<file>` tag syntax
app/model_catalog.py's CATALOG entries already use) — kept as a separate module from ollama_admin.py regardless,
same reasoning as app/services/matricxon_client.py's own docstring.

`pull_model_stream` branches on whether Matricxon is running locally: when it is,
app.services.matricxon_direct_puller.MatricxonDirectPuller downloads straight from Hugging Face instead of
proxying the whole (often multi-GB) download through Matricxon's own `/api/pull` — reported live (2026-09-21)
as "slowing it very much". A remote Matricxon host keeps using this file's own proxy-through-`/api/pull`
implementation below unchanged, since pAIring and a remote host don't share a filesystem to write straight to
disk into (see matricxon_direct_puller's own module docstring for that constraint).
"""

import json

import httpx

from app.services import matricxon_pool
from app.services.matricxon_client import MatricxonError
from app.services.matricxon_direct_puller import MatricxonDirectPuller

_REQUEST_TIMEOUT = httpx.Timeout(120.0, connect=5.0)
_PULL_TIMEOUT = httpx.Timeout(None, connect=5.0)


def _primary_host() -> str:
    return matricxon_pool.get_effective_hosts()[0]


def _is_local() -> bool:
    return matricxon_pool.get_effective_hosts() == [matricxon_pool.LOCAL_MATRICXON_HOST]


async def pull_model_stream(tag: str):
    """Downloads a model into Matricxon, yielding progress updates as they arrive — same NDJSON pull-progress
    shape as app.services.ollama_admin.pull_model_stream, forwarded the same way over SSE by the caller (see
    app/routers/settings.py). See this module's own docstring for the local-vs-remote dispatch."""
    if _is_local():
        async for progress in MatricxonDirectPuller().pull_stream(tag):
            yield progress
        return

    async with httpx.AsyncClient(timeout=_PULL_TIMEOUT) as client:
        try:
            async with client.stream(
                "POST", f"{_primary_host()}/api/pull", json={"model": tag, "stream": True}
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        raise MatricxonError(data["error"])
                    yield data
        except httpx.HTTPError as exc:
            raise MatricxonError(f"Pulling {tag} failed: {exc}") from exc


async def delete_model(tag: str) -> None:
    """Removes a model Matricxon already has pulled — the inverse of pull_model_stream, mirrors
    app.services.ollama_admin.delete_model's identical DELETE-with-JSON-body shape."""
    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
        try:
            resp = await client.request("DELETE", f"{_primary_host()}/api/delete", json={"model": tag})
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise MatricxonError(f"Deleting {tag} failed: {exc}") from exc
