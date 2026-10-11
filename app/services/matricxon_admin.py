"""Model administration against Matricxon's primary host (like ollama_admin.py; see its docstring): pull and
delete for admin-only model management. Matricxon's /api/pull and /api/delete use Ollama's wire shape
(../matricxon/app/routers/pull_router.py|delete_router.py;../matricxon/app/pull/tag.py resolves the same
`hf.co/<repo>:<file>` tags as model_catalog.py). Separate from ollama_admin.py as in matricxon_client.py.

`pull_model_stream` branches on local vs remote: locally, MatricxonDirectPuller downloads straight from
Hugging Face instead of proxying the (often multi-GB) download through Matricxon's `/api/pull`, reported
as "slowing it very much". A remote host keeps the proxy-through-`/api/pull` path below,
since pAIring and a remote host share no filesystem (see matricxon_direct_puller).
"""

import json

import httpx

from app.services import matricxon_http, matricxon_pool
from app.services.matricxon_client import MatricxonError
from app.services.matricxon_direct_puller import MatricxonDirectPuller

_REQUEST_TIMEOUT = httpx.Timeout(120.0, connect=5.0)
_PULL_TIMEOUT = httpx.Timeout(None, connect=5.0)


def _primary_host() -> str:
    return matricxon_pool.get_effective_hosts()[0]


async def _model_host() -> str:
    """The node that changes the models. In a cluster the models folder is shared, so one node at a time does it:
    the model-lease owner (needs a quorum). Alone, the primary host.
    """
    running, holder = await matricxon_pool.lease_holder("model_pull")
    if holder:
        return holder
    if running:
        raise MatricxonError(
            "No node owns the cluster's model lease right now (the cluster has no quorum). "
            "Models can be added or removed again once it does."
        )
    return _primary_host()


def _is_local() -> bool:
    return matricxon_pool.get_effective_hosts() == [matricxon_pool.LOCAL_MATRICXON_HOST]


async def pull_model_stream(tag: str):
    """Downloads a model into Matricxon, yielding progress in the same NDJSON shape as
    ollama_admin.pull_model_stream (forwarded as SSE by routers/settings.py). See the module docstring for local vs
    remote.
    """
    if _is_local():
        async for progress in MatricxonDirectPuller().pull_stream(tag):
            yield progress
        return

    host = await _model_host()
    async with matricxon_http.client(timeout=_PULL_TIMEOUT) as client:
        try:
            async with client.stream("POST", f"{host}/api/pull", json={"model": tag, "stream": True}) as resp:
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
    """Removes a model Matricxon has; the inverse of pull_model_stream (same DELETE-with-JSON-body as
    ollama_admin.delete_model)."""
    host = await _model_host()
    async with matricxon_http.client(timeout=_REQUEST_TIMEOUT) as client:
        try:
            resp = await client.request("DELETE", f"{host}/api/delete", json={"model": tag})
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise MatricxonError(f"Deleting {tag} failed: {exc}") from exc
