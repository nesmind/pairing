"""A GitHub repo's tag list for Settings > Inference engines' version picker (shared available-versions
endpoint in routers/ollama_admin, matricxon_admin, sdcpp_admin). Helps fill the installers' install_repo/
install_version overrides correctly (a typo'd tag otherwise fails several steps into an install).

GitHub's unauthenticated REST API allows 60 requests/hour per IP, easily exhausted, so results are cached
per repo for _CACHE_TTL_SECONDS. Any failure (network, 404 typo'd repo, rate limit) gives an empty list
rather than raising: a convenience over free-text entry that must never block Settings.
"""

import logging
import time

import httpx

logger = logging.getLogger("llama_chat")

_REQUEST_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
_PER_PAGE = 30
_CACHE_TTL_SECONDS = 600.0

# repo -> (fetched_at, tag names newest-first). A module-level dict with a TTL check suffices (like
# matricxon_pool's cached host list); few distinct repos are ever used.
_cache: dict[str, tuple[float, list[str]]] = {}


async def list_tags(repo: str) -> list[str]:
    """Up to the most recent `_PER_PAGE` tag names for `repo` (e.g. "ollama/ollama"), newest first (GitHub's
    /tags order). Empty for a blank repo or any failure.
    """
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
