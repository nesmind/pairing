"""
Searches and verifies GGUF models on Hugging Face — the only source
Settings > Model's "Browse more models" now uses (see
app/services/extended_model_catalog_service.py), replacing an earlier
version that searched Ollama's own registry. Every field this module
returns comes straight from Hugging Face's real Hub API, confirmed
live rather than assumed:

- GET /api/models?search=<q>&filter=gguf — repo search, filtered to
  repos tagged "gguf" (the only ones Ollama can actually run).
- GET /api/models/<repo_id>?blobs=true — one call gets everything a
  repo's own /library/<model>/tags page would: every file's real byte
  size (no separate HEAD request per file needed, confirmed live), plus
  a `gguf` metadata block with real parameter count and context
  length — richer than Ollama's own registry ever exposed (that one
  had no context_length at all).

A model actually gets pulled through Ollama's own existing hf.co/
passthrough (`ollama pull hf.co/<repo_id>:<tag>`, already supported —
see app/model_catalog.py's MiniMax entry, added before this module
existed) — this file only searches/verifies; it never downloads model
weights itself. `<tag>` is built from the exact GGUF filename (minus
its extension) rather than a hand-extracted "quant code": Ollama
matches the tag against candidate filenames case-insensitively, so the
full filename is guaranteed to match unambiguously, where a heuristic
substring extraction could pick the wrong file on an unusually-named
repo.
"""

import httpx

HF_API_BASE = "https://huggingface.co/api/models"
_TIMEOUT = httpx.Timeout(15.0, connect=10.0)
_SEARCH_LIMIT = 20


class HuggingFaceLookupError(Exception):
    """Raised when a repo/search can't be reached or resolved — not found, or Hugging Face itself is
    unreachable (no internet, or a misconfigured/absent proxy). Same "turn a low-level failure into a clean,
    actionable message" convention as app.services.ollama_client.OllamaError."""


def _license(card_data: dict) -> str | None:
    return card_data.get("license_name") or card_data.get("license")


async def search_models(query: str, proxy_url: str | None) -> list[dict]:
    """Up to 20 GGUF-tagged repos matching `query`, ranked by download count — same ranking
    huggingface.co/models itself defaults to. Returns [] for a query with no matches (not an error — see
    huggingface.co's own behavior, confirmed live: an empty result set is just an empty JSON array, not a 404)."""
    async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True, proxy=proxy_url) as client:
        try:
            resp = await client.get(
                HF_API_BASE,
                params={
                    "search": query,
                    "filter": "gguf",
                    "sort": "downloads",
                    "direction": -1,
                    "limit": _SEARCH_LIMIT,
                },
            )
        except httpx.HTTPError as exc:
            raise HuggingFaceLookupError(
                f'Could not reach Hugging Face to search for "{query}": {exc}. Check this machine\'s internet '
                "connection, or configure an outbound proxy under Settings > System."
            ) from exc
        if resp.status_code != 200:
            raise HuggingFaceLookupError(f"Hugging Face returned an unexpected error ({resp.status_code}).")
        repos = resp.json()

    return [
        {
            "repo_id": repo["id"],
            "downloads": repo.get("downloads", 0),
            "likes": repo.get("likes", 0),
            "gated": bool(repo.get("gated")),
            "license": None,  # search results don't include cardData — only the per-repo lookup below does
        }
        for repo in repos
    ]


# GGUF files this app can't sensibly offer as a single pullable tag — sharded multi-part weights (e.g.
# "-00001-of-00002.gguf") would need every shard pulled together, which Ollama's hf.co/ single-tag passthrough
# has no way to express; skipped rather than shown as if they were a normal, complete, single-file option.
def _is_single_file_gguf(filename: str) -> bool:
    return filename.endswith(".gguf") and "-of-" not in filename


async def get_repo_files(repo_id: str, proxy_url: str | None) -> dict:
    """Real metadata plus every single-file GGUF variant `repo_id` actually offers, each with its real download
    size — the "Browse more models" file-picker step, shown once an admin picks a repo from search_models'
    results (or already knows the exact repo path and skips search). Raises HuggingFaceLookupError if the repo
    doesn't exist (Hugging Face's own API returns 401, not 404, for an unauthenticated request to a nonexistent
    repo — confirmed live, not assumed — so both are treated as "not found" here) or Hugging Face is
    unreachable."""
    async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True, proxy=proxy_url) as client:
        try:
            resp = await client.get(f"{HF_API_BASE}/{repo_id}", params={"blobs": "true"})
        except httpx.HTTPError as exc:
            raise HuggingFaceLookupError(
                f'Could not reach Hugging Face to look up "{repo_id}": {exc}. Check this machine\'s internet '
                "connection, or configure an outbound proxy under Settings > System."
            ) from exc
        if resp.status_code in (401, 404):
            raise HuggingFaceLookupError(f'"{repo_id}" was not found on Hugging Face — check the repo is correct.')
        if resp.status_code != 200:
            raise HuggingFaceLookupError(
                f'Hugging Face returned an unexpected error for "{repo_id}" ({resp.status_code}).'
            )
        repo = resp.json()

    gguf_meta = repo.get("gguf") or {}
    files = [
        {"filename": s["rfilename"], "download_gb": round(s["size"] / 1_000_000_000, 2)}
        for s in repo.get("siblings", [])
        if _is_single_file_gguf(s["rfilename"]) and s.get("size")
    ]
    return {
        "repo_id": repo_id,
        "family": gguf_meta.get("architecture"),
        "parameter_size": _format_param_count(gguf_meta.get("total")),
        "context_length": gguf_meta.get("context_length"),
        "gated": bool(repo.get("gated")),
        "license": _license(repo.get("cardData") or {}),
        "files": files,
    }


def _format_param_count(total: int | None) -> str | None:
    """1_235_814_432 -> "1.2B" — huggingface.co's own gguf.total is a raw parameter count, not the
    human-readable "N.NB" string app/model_catalog.py's hand-curated entries and Ollama's registry both use;
    formatted here once so every caller (the catalog UI, hardware-gating display) sees the same convention
    regardless of where an entry came from."""
    if not total:
        return None
    if total >= 1_000_000_000:
        return f"{total / 1_000_000_000:.1f}B"
    return f"{total / 1_000_000:.0f}M"
