"""Searches and verifies GGUF models on Hugging Face, the only source of Settings > Model's "Browse more
models" (extended_model_catalog_service.py). Fields come from the real Hub API:

- GET /api/models?search=<q>&filter=gguf: repo search for "gguf"-tagged repos.
- GET /api/models/<repo_id>?blobs=true: each file's byte size plus a `gguf` block with parameter count and context
  length.

Pulls go through Ollama's hf.co/ passthrough (`ollama pull hf.co/<repo_id>:<tag>`) or Matricxon's
equivalent; this file never downloads weights. `<tag>` is the exact GGUF filename minus extension, not a
quant code: both engines match tags to filenames case-insensitively, so the full name is unambiguous where
substring extraction could pick the wrong file.
"""

import httpx

HF_API_BASE = "https://huggingface.co/api/models"
_TIMEOUT = httpx.Timeout(15.0, connect=10.0)
_SEARCH_LIMIT = 20


class HuggingFaceLookupError(Exception):
    """Raised when a repo/search can't be reached or resolved (not found, or Hugging Face unreachable: no internet or
    bad proxy). A clean message, like ollama_client.OllamaError."""


class HuggingFaceCatalogSearch:
    """The "Browse more models" search and file-picker steps. Results are read-only and never trusted for storage:
    ExtendedModelCatalog.add re-verifies via repo_files.
    """

    @staticmethod
    async def search(query: str, proxy_url: str | None) -> list[dict]:
        """Up to 20 GGUF-tagged repos matching `query`, by downloads (huggingface.co/models' default). [] for no
        matches (an empty array, not a 404).
        """
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
                "is_image": HuggingFaceCatalogSearch.is_image_repo(repo),
                # Search results lack cardData; only repo_files has it.
                "license": None,
            }
            for repo in repos
        ]

    @staticmethod
    async def repo_files(repo_id: str, proxy_url: str | None) -> dict:
        """Metadata plus every single-file GGUF variant of `repo_id` with real download sizes: the file-picker step.
        Raises HuggingFaceLookupError if the repo doesn't exist (the unauthenticated API returns 401, not 404; both
        mean "not found") or Hugging Face is unreachable.
        """
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
        # A same-repo mmproj sidecar makes every non-projector file a vision model (Ollama auto-pairs it on hf.co/
        # pull, Matricxon pulls it alongside; see model_catalog_service's curated `vision` flag), so the picker badges
        # "+Vision".
        has_projector = any(HuggingFaceCatalogSearch.is_projector_file(s["rfilename"]) for s in gguf_siblings)
        files = [
            {
                "filename": s["rfilename"],
                "download_gb": round(s["size"] / 1_000_000_000, 2),
                # See HuggingFaceCatalogSearch.is_projector_file: when True, family/parameter_size rarely describe this
                # file, so the picker flags it rather than attributing the main model's architecture.
                "is_projector": HuggingFaceCatalogSearch.is_projector_file(s["rfilename"]),
                "vision": has_projector and not HuggingFaceCatalogSearch.is_projector_file(s["rfilename"]),
                # Git-LFS sha256 and exact byte size (download_gb is too rounded to verify a download), on every GGUF
                # sibling from this same call. Internal only (not in HfFileOption; pydantic drops it).
                # MatricxonDirectPuller verifies direct downloads against them like Matricxon's HFDownloader.
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
            "is_image": HuggingFaceCatalogSearch.is_image_repo(repo),
            "license": HuggingFaceCatalogSearch._license(repo.get("cardData") or {}),
            "files": files,
        }

    # Hub tags marking a text-to-image repo: it can't run on a chat engine (no `general.architecture`) and belongs
    # to the image engine's store (image_model_service). "image-text-to-text" (a vision *chat* model) is not here.
    _IMAGE_PIPELINE_TAGS = frozenset({"text-to-image", "image-to-image"})
    _IMAGE_REPO_TAGS = frozenset({"sd.cpp", "stable-diffusion.cpp"})

    @staticmethod
    def is_image_repo(repo: dict) -> bool:
        """True for a diffusion repo, from `pipeline_tag`/`tags` (in search and per-repo lookups). Some sd.cpp
        conversions lack a pipeline_tag, so stable-diffusion.cpp community tags count too.
        """
        return repo.get("pipeline_tag") in HuggingFaceCatalogSearch._IMAGE_PIPELINE_TAGS or bool(
            HuggingFaceCatalogSearch._IMAGE_REPO_TAGS & set(repo.get("tags") or [])
        )

    # GGUF files not offerable as one pullable tag: sharded weights ("-00001-of-00002.gguf") need all shards,
    # which neither engine's hf.co/ single-tag passthrough expresses. Skipped, not shown as a normal option.
    @staticmethod
    def _is_single_file_gguf(filename: str) -> bool:
        return filename.endswith(".gguf") and "-of-" not in filename

    # A vision-projector (mmproj/CLIP) sidecar, not a chat model (the naming convention of model_catalog.py's
    # vision entries). A multi-file repo's one `gguf` block describes Hugging Face's primary file, almost never
    # a projector, so per-file choices need this to know what that block covers (see repo_files).
    @staticmethod
    def is_projector_file(filename: str) -> bool:
        return "mmproj" in filename.lower()

    @staticmethod
    def _format_param_count(total: int | None) -> str | None:
        """1_235_814_432 -> "1.2B": Hugging Face's gguf.total is a raw count, unlike the "N.NB" strings in
        model_catalog.py and Ollama's registry.
        """
        if not total:
            return None
        if total >= 1_000_000_000:
            return f"{total / 1_000_000_000:.1f}B"
        return f"{total / 1_000_000:.0f}M"

    @staticmethod
    def _license(card_data: dict) -> str | None:
        return card_data.get("license_name") or card_data.get("license")
