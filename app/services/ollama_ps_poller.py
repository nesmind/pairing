"""Polls Ollama's GET /api/ps ("what's loaded now") on every host and appends a snapshot row per (host, model)
(read back by ollama_snapshot_service for the Telemetry page). The first periodic loop in this codebase;
every other asyncio.create_task is a one-shot per request/event.

Primary-gated (startup_service.run_startup_tasks): loaded models are host-level state, so every instance polling
would only duplicate rows.
"""

import asyncio
import logging
from datetime import UTC, datetime

import httpx

from app.database import AsyncSessionLocal
from app.models import ModelSnapshot
from app.services import ollama_pool

logger = logging.getLogger("llama_chat")

POLL_INTERVAL_SECONDS = 30.0
_PS_TIMEOUT = httpx.Timeout(5.0, connect=2.0)

# Same strong-ref set/done-callback idiom as title_service._background_title_tasks: a bare create_task
# result is weakly referenced and could be garbage-collected mid-run.
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
            engine="ollama",
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
    for host in ollama_pool.get_effective_hosts():
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


async def _run_ollama_ps_poller() -> None:
    while True:
        try:
            await _poll_once()
        except Exception:
            # A bad poll (e.g. a transient DB error) must never kill the loop for good.
            logger.exception("Ollama /api/ps poll failed.")
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


def start_ollama_ps_poller() -> None:
    """Called once from startup_service.run_startup_tasks (primary only); see module docstring."""
    task = asyncio.create_task(_run_ollama_ps_poller())
    _background_poller_tasks.add(task)
    task.add_done_callback(_background_poller_tasks.discard)
