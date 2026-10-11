"""Polls Matricxon's GET /api/ps ("what's loaded now") on every host and appends a snapshot row per (host,
model): the twin of ollama_ps_poller (see its docstring for why a poller, primary-gated). Field-compatible
with../matricxon/app/schemas/ps.py (name/size/size_vram/expires_at, as _poll_host parses): /api/ps is
Ollama-compatible like every endpoint matricxon_client.py uses.
"""

import asyncio
import logging
from datetime import UTC, datetime

import httpx

from app.database import AsyncSessionLocal
from app.models import ModelSnapshot
from app.services import matricxon_pool

logger = logging.getLogger("llama_chat")

POLL_INTERVAL_SECONDS = 30.0
_PS_TIMEOUT = httpx.Timeout(5.0, connect=2.0)

# Same strong-ref set/done-callback idiom as ollama_ps_poller: a bare create_task result is weakly
# referenced and could be garbage-collected mid-run.
_background_poller_tasks: set[asyncio.Task] = set()


def _parse_expires_at(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw).astimezone(UTC)
    except ValueError:
        return None


async def _poll_host(host: str) -> list[ModelSnapshot]:
    polled_at = datetime.now(UTC)
    async with httpx.AsyncClient(timeout=_PS_TIMEOUT) as client:
        resp = await client.get(f"{host}/api/ps")
        resp.raise_for_status()
    return [
        ModelSnapshot(
            engine="matricxon",
            host=host,
            model_name=m.get("name") or m.get("model", "unknown"),
            size_bytes=m.get("size"),
            size_vram_bytes=m.get("size_vram"),
            expires_at=_parse_expires_at(m.get("expires_at")),
            polled_at=polled_at,
        )
        for m in resp.json().get("models", [])
    ]


async def _poll_once() -> None:
    snapshots: list[ModelSnapshot] = []
    for host in matricxon_pool.get_effective_hosts():
        try:
            snapshots.extend(await _poll_host(host))
        except httpx.HTTPError as exc:
            # One unreachable host mustn't stop the others being polled.
            logger.warning("Could not poll /api/ps on %s: %s", host, exc)
    if not snapshots:
        return
    async with AsyncSessionLocal() as db:
        db.add_all(snapshots)
        await db.commit()


async def _run_matricxon_ps_poller() -> None:
    while True:
        try:
            await _poll_once()
        except Exception:
            # A bad poll (e.g. a transient DB error) must never kill the loop for good.
            logger.exception("Matricxon /api/ps poll failed.")
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


def start_matricxon_ps_poller() -> None:
    """Called once from startup_service.run_startup_tasks (primary only); see ollama_ps_poller."""
    task = asyncio.create_task(_run_matricxon_ps_poller())
    _background_poller_tasks.add(task)
    task.add_done_callback(_background_poller_tasks.discard)
