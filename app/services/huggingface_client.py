"""
Searches and verifies GGUF models on Hugging Face — the only source Settings > Model's "Browse more models"
uses (see app/services/extended_model_catalog_service.py). Every field this module returns comes straight from
Hugging Face's real Hub API, confirmed live rather than assumed:

- GET /api/models?search=<q>&filter=gguf — repo search, filtered to repos tagged "gguf".
- GET /api/models/<repo_id>?blobs=true — every file's real byte size plus a `gguf` metadata block with real
  parameter count and context length.

A model actually gets pulled through Ollama's own existing hf.co/ passthrough (`ollama pull
hf.co/<repo_id>:<tag>`) or Matricxon's equivalent — this file only searches/verifies, it never downloads model
weights itself. `<tag>` is built from the exact GGUF filename (minus its extension) rather than a
hand-extracted "quant code": both engines match the tag against candidate filenames case-insensitively, so the
full filename is guaranteed to match unambiguously, where a heuristic substring extraction could pick the wrong
file on an unusually-named repo.
"""

import httpx

HF_API_BASE = "https://huggingface.co/api/models"
_TIMEOUT = httpx.Timeout(15.0, connect=10.0)
_SEARCH_LIMIT = 20


class HuggingFaceLookupError(Exception):
    """Raised when a repo/search can't be reached or resolved — not found, or Hugging Face itself is
    unreachable (no internet, or a misconfigured/absent proxy). Same "turn a low-level failure into a clean,
    actionable message" convention as app.services.ollama_client.OllamaError."""


class HuggingFaceCatalogSearch:
    """The "Browse more models" search + file-picker steps — a search result/repo listing is read-only and
    never itself trusted for what gets stored (see extended_model_catalog_service.ExtendedModelCatalog.add,
    which re-verifies from scratch via repo_files rather than trusting a caller's earlier search result)."""

    @staticmethod
    async def search(query: str, proxy_url: str | None) -> list[dict]:
        """Up to 20 GGUF-tagged repos matching `query`, ranked by download count — same ranking
        huggingface.co/models itself defaults to. Returns [] for a query with no matches (not an error — an
        empty result set is just an empty JSON array, not a 404, confirmed live)."""
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
                # Search results don't include cardData — only repo_files below does.
                "license": None,
            }
            for repo in repos
        ]

    @staticmethod
    async def repo_files(repo_id: str, proxy_url: str | None) -> dict:
        """Real metadata plus every single-file GGUF variant `repo_id` actually offers, each with its real
        download size — the file-picker step shown once an admin picks a repo from search()'s own results (or
        already knows the exact repo path and skips search). Raises HuggingFaceLookupError if the repo doesn't
        exist (Hugging Face's own API returns 401, not 404, for an unauthenticated request to a nonexistent
        repo — confirmed live, both are treated as "not found" here) or Hugging Face is unreachable."""
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True, proxy=proxy_url) as client:
            try:
                resp = await client.get(f"{HF_API_BASE}/{repo_id}", params={"blobs": "true"})
            except httpx.HTTPError as exc:
                raise HuggingFaceLookupError(
                    f'Could not reach Hugging Face to look up "{repo_id}": {exc}. Check this machine\'s '
                    "internet connection, or configure an outbound proxy under Settings > System."
                ) from exc
            if resp.status_code in (401, 404):
                raise HuggingFaceLookupError(f'"{repo_id}" was not found on Hugging Face — check the repo is correct.')
            if resp.status_code != 200:
                raise HuggingFaceLookupError(
                    f'Hugging Face returned an unexpected error for "{repo_id}" ({resp.status_code}).'
                )
            repo = resp.json()

        gguf_meta = repo.get("gguf") or {}
        gguf_siblings = [
            s
            for s in repo.get("siblings", [])
            if HuggingFaceCatalogSearch._is_single_file_gguf(s["rfilename"]) and s.get("size")
        ]
        # A same-repo mmproj sidecar means every non-projector file here is a vision model — Ollama auto-pairs
        # that sidecar on a hf.co/ pull and Matricxon pulls it alongside (see model_catalog_service's own
        # comment on the curated `vision` flag), so the file picker can badge them "+Vision" upfront.
        has_projector = any(HuggingFaceCatalogSearch.is_projector_file(s["rfilename"]) for s in gguf_siblings)
        files = [
            {
                "filename": s["rfilename"],
                "download_gb": round(s["size"] / 1_000_000_000, 2),
                # See HuggingFaceCatalogSearch.is_projector_file's own docstring: this repo's family/
                # parameter_size below almost never describes this specific file when True — the file-picker UI
                # flags it instead of silently attributing the main model's architecture to a projector sidecar.
                "is_projector": HuggingFaceCatalogSearch.is_projector_file(s["rfilename"]),
                "vision": has_projector and not HuggingFaceCatalogSearch.is_projector_file(s["rfilename"]),
                # Real git-LFS sha256 and exact byte size (download_gb above is rounded, too imprecise to
                # compare a completed download against) — both already present on every real GGUF sibling
                # (always LFS-tracked by their size alone), known upfront from this same call, no extra
                # request. Internal-only today (not in HfFileOption/the API response, silently dropped by
                # pydantic's default "ignore extra fields" behavior):
                # app.services.matricxon_direct_puller.MatricxonDirectPuller verifies a direct-from-HF
                # download against these the same way Matricxon's own HFDownloader already does for its own
                # pulls.
                "sha256": s.get("lfs", {}).get("sha256"),
                "size_bytes": s["size"],
            }
            for s in gguf_siblings
        ]
        return {
            "repo_id": repo_id,
            "family": gguf_meta.get("architecture"),
            "parameter_size": HuggingFaceCatalogSearch._format_param_count(gguf_meta.get("total")),
            "context_length": gguf_meta.get("context_length"),
            "gated": bool(repo.get("gated")),
            "license": HuggingFaceCatalogSearch._license(repo.get("cardData") or {}),
            "files": files,
        }

    # GGUF files this app can't sensibly offer as a single pullable tag — sharded multi-part weights (e.g.
    # "-00001-of-00002.gguf") would need every shard pulled together, which neither engine's hf.co/ single-tag
    # passthrough has a way to express; skipped rather than shown as if they were a normal, complete option.
    @staticmethod
    def _is_single_file_gguf(filename: str) -> bool:
        return filename.endswith(".gguf") and "-of-" not in filename

    # A vision-projector (mmproj/CLIP) sidecar, not a standalone chat model — same llama.cpp/GGUF-ecosystem
    # naming convention app/model_catalog.py's own hand-curated vision entries already rely on. A multi-file
    # repo's *one* repo-level `gguf` metadata block (see repo_files' own docstring) describes whichever file
    # Hugging Face picked as primary — almost never a same-repo projector sidecar — so a caller offering
    # per-file choices from that same repo needs this to know which files that block actually describes.
    @staticmethod
    def is_projector_file(filename: str) -> bool:
        return "mmproj" in filename.lower()

    @staticmethod
    def _format_param_count(total: int | None) -> str | None:
        """1_235_814_432 -> "1.2B" — huggingface.co's own gguf.total is a raw parameter count, not the
        human-readable "N.NB" string app/model_catalog.py's hand-curated entries and Ollama's registry both
        use."""
        if not total:
            return None
        if total >= 1_000_000_000:
            return f"{total / 1_000_000_000:.1f}B"
        return f"{total / 1_000_000:.0f}M"

    @staticmethod
    def _license(card_data: dict) -> str | None:
        return card_data.get("license_name") or card_data.get("license")
