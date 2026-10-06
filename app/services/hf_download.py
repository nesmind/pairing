"""Streams one Hugging Face file to disk (via a .part file) as {"status","completed","total"} progress events."""

from collections.abc import AsyncIterator
from pathlib import Path

import httpx

_TIMEOUT = httpx.Timeout(None, connect=10.0)


async def stream_file(url: str, dest: Path, proxy_url: str | None, label: str) -> AsyncIterator[dict]:
    """Yields progress events; a failure yields a final {"error": ...} (and leaves no partial file)."""
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True, proxy=proxy_url) as client:
            async with client.stream("GET", url) as resp:
                if resp.status_code in (401, 403):
                    yield {"error": f"{label} is gated on Hugging Face — it can't be downloaded without a token."}
                    return
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0)) or None
                completed, last_pct = 0, -1
                with open(part, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)
                        completed += len(chunk)
                        if total and completed * 100 // total != last_pct:
                            last_pct = completed * 100 // total
                            yield {"status": f"Downloading {label}", "completed": completed, "total": total}
        part.replace(dest)
    except (httpx.HTTPError, OSError) as exc:
        part.unlink(missing_ok=True)
        yield {"error": f"Could not download {label}: {exc}"}
