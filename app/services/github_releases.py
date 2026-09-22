"""Fetches a GitHub repo's real tag list for Settings > External servers' version picker (see
app.routers.ollama_admin/comfyui_admin/matricxon_admin's shared available-versions endpoint, and
app.services.ollama_installer/comfyui_installer/matricxon_installer, whose own install_repo/install_version
override this is meant to make easier to fill in correctly — a tag typo today only surfaces as a failed install
several steps in, not a validation error up front).

GitHub's REST API is unauthenticated here (no token configured anywhere in this app) and rate-limited to 60
requests/hour per source IP — trivial to exhaust with more than a couple of admins loading Settings, so results
are cached in-process per repo for _CACHE_TTL_SECONDS. Any failure (network, 404 for a typo'd repo, rate limit)
degrades to an empty list rather than raising — this is a convenience on top of the free-text override field,
never something that should block Settings from loading or an admin from typing a tag by hand.
"""

import logging
import time

import httpx

logger = logging.getLogger("llama_chat")

_REQUEST_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
_PER_PAGE = 30
_CACHE_TTL_SECONDS = 600.0

# repo -> (fetched_at, tag names newest-first) — a plain module-level dict is enough here, same "no real
# eviction, just a TTL check on read" idiom as app.services.matricxon_pool's own cached host list; the number of
# distinct repos an admin could plausibly point this at in one process's lifetime is tiny.
_cache: dict[str, tuple[float, list[str]]] = {}


async def list_tags(repo: str) -> list[str]:
    """Up to the most recent `_PER_PAGE` tag names for `repo` (e.g. "ollama/ollama"), newest first — GitHub's
    own /tags endpoint already returns them in that order. Empty list for an empty/blank repo, or any failure."""
    if not repo:
        return []
    cached = _cache.get(repo)
    if cached is not None and (time.monotonic() - cached[0]) < _CACHE_TTL_SECONDS:
        return cached[1]

    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            resp = await client.get(
                f"https://api.github.com/repos/{repo}/tags",
                params={"per_page": _PER_PAGE},
                headers={"Accept": "application/vnd.github+json"},
            )
            resp.raise_for_status()
            tags = [entry["name"] for entry in resp.json()]
    except httpx.HTTPError as exc:
        logger.warning("Could not fetch GitHub tags for %s: %s", repo, exc)
        return cached[1] if cached is not None else []

    _cache[repo] = (time.monotonic(), tags)
    return tags
