"""Model administration against Ollama's "primary" host (the first of ollama_pool.get_effective_hosts: local,
or the first remote): pull and delete for admin-only model management. Split from ollama_client.py (inference,
routed across the pool) for the file-size rule.

Not routed across the pool: installed models aren't guaranteed uniform across instances, so pull/delete
target one explicit host; mirroring to the rest is manual for now.
"""

import json

import httpx

from app.services import ollama_pool
from app.services.ollama_client import OllamaError

# Same values as ollama_client.py's, duplicated rather than importing its private constants for two httpx.Timeouts.
_REQUEST_TIMEOUT = httpx.Timeout(120.0, connect=5.0)
_PULL_TIMEOUT = httpx.Timeout(None, connect=5.0)


def _primary_host() -> str:
    return ollama_pool.get_effective_hosts()[0]


async def pull_model_stream(tag: str):
    """Downloads a model into Ollama, yielding Ollama's NDJSON pull progress (`status`, usually `completed`/`total`).
    routers/settings.py forwards it as SSE so "Pull" shows a progress bar during multi-GB downloads.
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
    """Removes a pulled model, freeing disk (admin "Uninstall", routers/settings.py); the inverse of pull_model_stream.
    Ollama's delete takes DELETE with a JSON body, avoiding URL-encoding tags with slashes (e.g.
    `hf.co/unsloth/MiniMax-M3-GGUF:...`).
    """
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
