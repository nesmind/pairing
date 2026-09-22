"""
Downloads a Matricxon model directly from Hugging Face instead of proxying the whole download through
Matricxon's own `/api/pull` (app.services.matricxon_admin's other implementation, still used for a remote
Matricxon host — see that module's own dispatch) — the proxy-through-Matricxon path was reported live
(2026-09-21) as "slowing it very much" on a local install. This writes the `.gguf` and its matching sidecar
straight into Matricxon's own models directory using the same real Hugging Face APIs "Browse more models"
already relies on (app.services.huggingface_client, app.services.gguf_probe), so Matricxon is never involved
in the download itself — only in the one small `/api/infer-capabilities` call for the one piece of logic
(`CapabilityInferer`) that would otherwise have to be reimplemented here and risk drifting from Matricxon's
own real rules.

Only usable when Matricxon is running locally (app.services.matricxon_admin's own dispatch decides this) —
pAIring and Matricxon must share a filesystem for writing straight to disk to make any sense at all.

Known, accepted limitations (not solved here, same "best effort within this feature's own scope" as every
other real gap surfaced this session):
  - `context_length` in the sidecar comes from the repo-level `gguf` metadata block
    (HuggingFaceCatalogSearch.repo_files), not a real per-file read the way `architecture` is (see
    gguf_probe's own module docstring on why that block can misattribute *architecture* for a non-primary
    file) — an accepted, purely cosmetic approximation, not worth a riskier extension of that hand-rolled
    binary parser for a display-only field.
  - Doesn't honor Settings > System's outbound-proxy config (app.services.http_proxy_service) — the
    proxy-through-Matricxon path it replaces never did either (Matricxon downloaded on its own network, with
    its own environment's proxy settings, unrelated to pAIring's own), so this is parity, not a regression.
"""

import hashlib
import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx

from app.database import AsyncSessionLocal
from app.schemas import MatricxonServerConfig
from app.services import gguf_probe, matricxon_pool, matricxon_process, settings_service
from app.services.download_coordinator import get_download_coordinator
from app.services.huggingface_client import HuggingFaceCatalogSearch
from app.services.matricxon_client import MatricxonError

_TIMEOUT = httpx.Timeout(None, connect=10.0)
_CHUNK_SIZE = 1024 * 1024
# Byte-based rather than Matricxon's own time-based throttle (../matricxon/app/pull/downloader.py) — this
# module has no other reason to import `time`, and a fixed byte interval is just as effective at keeping the
# SSE stream from flooding the browser with a progress line per 1MB chunk.
_PROGRESS_THROTTLE_BYTES = 8 * 1024 * 1024


class MatricxonDirectPuller:
    async def pull_stream(self, tag: str) -> AsyncIterator[dict]:
        """Same NDJSON pull-progress shape app.services.matricxon_admin.pull_model_stream already yields
        (`status`, plus `completed`/`total` while downloading) — app/routers/settings.py's `event_stream`
        forwards whatever this yields unchanged, so it needs no changes either way. Raises MatricxonError on
        any failure, the same type the proxy-through-Matricxon path already raises, so
        app.services.inference_client's existing `except (OllamaError, MatricxonError)` wrapping into
        InferenceError keeps working unchanged too."""
        repo_id, suffix = self._parse_tag(tag)
        config = await self._get_config()

        yield {"status": "resolving manifest"}
        matched, context_length = await self._resolve(repo_id, suffix)
        filename = matched["filename"]

        dest = self._resolve_dest(repo_id, filename, config)
        partial_path = dest.with_suffix(dest.suffix + ".partial")

        async with get_download_coordinator().guard(dest):
            if dest.exists():
                size = dest.stat().st_size
                yield {"status": "already downloaded", "completed": size, "total": size}
            else:
                async for progress in self._download(repo_id, matched, dest, partial_path):
                    yield progress

        architecture, parameter_count = await self._probe(repo_id, filename)
        capabilities = await self._infer_capabilities(repo_id, filename, architecture)
        self._write_sidecar(dest, tag, architecture, parameter_count, capabilities, context_length)
        yield {"status": "success"}

    @staticmethod
    def _parse_tag(tag: str) -> tuple[str, str]:
        """`tag` is always `hf.co/<repo_id>:<suffix>` — `suffix` is the *full* filename stem for every tag
        this app itself builds (app.services.extended_model_catalog_service.ExtendedModelCatalog.build_tag),
        but a hand-curated default_models.json entry can instead use a short, human-friendly one (e.g.
        "Q8_0" for a real file named "all-MiniLM-L6-v2.Q8_0.gguf" — confirmed live, 2026-09-21, this exact
        tag) — see _match's own docstring for how that gets resolved to a real filename either way."""
        if not tag.startswith("hf.co/") or ":" not in tag:
            raise MatricxonError(f'"{tag}" is not an hf.co/<repo>:<file> tag — cannot pull it directly.')
        repo_id, _, suffix = tag.removeprefix("hf.co/").partition(":")
        return repo_id, suffix

    @staticmethod
    async def _get_config() -> MatricxonServerConfig:
        """Self-contained fresh-session fetch — this call chain (routers/settings.py's pull_model ->
        inference_client.pull_model_stream's engine-agnostic dispatch -> matricxon_admin.pull_model_stream)
        has no request-scoped `db` reachable this deep without threading it through that dispatch layer for
        one engine-specific need, so this fetches its own, same pattern
        reply_termination_service._persist_cancelled_reply already uses."""
        async with AsyncSessionLocal() as db:
            return await settings_service.get_matricxon_server_config(db)

    @staticmethod
    def _resolve_dest(repo_id: str, filename: str, config: MatricxonServerConfig) -> Path:
        if config.models_path:
            return Path(config.models_path) / "hf.co" / repo_id / filename
        project_dir = matricxon_process.resolve_project_dir(config.project_dir)
        if project_dir is None:
            raise MatricxonError("Matricxon's local install couldn't be found to write the pulled model into.")
        return project_dir / "data" / "models" / "hf.co" / repo_id / filename

    @staticmethod
    async def _resolve(repo_id: str, suffix: str) -> tuple[dict, int | None]:
        """Never trusts a stale caller-provided size/sha256 — re-fetches the real repo listing, same "verify
        from scratch" philosophy app.services.extended_model_catalog_service.ExtendedModelCatalog.add already
        established for admin-added entries. Returns the matched file dict plus the repo-level context_length
        (see this module's own docstring on why that one field is an approximation)."""
        try:
            repo = await HuggingFaceCatalogSearch.repo_files(repo_id, proxy_url=None)
        except Exception as exc:  # noqa: BLE001 - HuggingFaceLookupError or a raw httpx failure either way
            raise MatricxonError(
                f'Could not resolve "{repo_id}" on Hugging Face: {MatricxonDirectPuller._describe(exc)}'
            ) from exc
        matched = MatricxonDirectPuller._match(suffix, repo["files"])
        return matched, repo["context_length"]

    @staticmethod
    def _describe(exc: Exception) -> str:
        """httpx (and some stdlib) exceptions can stringify to an empty message — confirmed live, 2026-09-22:
        a real transient failure surfaced as "download of ... failed: " with nothing after the colon, giving
        no clue what actually went wrong. Falls back to the exception's own type name, plus the request URL
        when httpx attached one, so a caller always has something real to act on."""
        message = str(exc)
        if message:
            return message
        request = getattr(exc, "request", None)
        if request is not None:
            return f"{type(exc).__name__} for {request.url}"
        return type(exc).__name__

    @staticmethod
    def _match(suffix: str, files: list[dict]) -> dict:
        """Same exact-stem-then-substring-fallback algorithm ../matricxon/app/pull/hf_resolver.py's own
        HFRepoResolver._match already uses, so a hand-curated tag with a short suffix (see _parse_tag's own
        docstring — confirmed live, 2026-09-21, against a real MiniLM tag whose suffix is "Q8_0" but whose
        real filename is "all-MiniLM-L6-v2.Q8_0.gguf") still resolves correctly here, not just for tags this
        app itself builds with the full stem already."""
        suffix_lower = suffix.lower()
        exact = [f for f in files if Path(f["filename"]).stem.lower() == suffix_lower]
        if len(exact) == 1:
            return exact[0]
        substring = [f for f in files if suffix_lower in f["filename"].lower()]
        if len(substring) == 1:
            return substring[0]
        candidates = ", ".join(f["filename"] for f in files)
        if not exact and not substring:
            raise MatricxonError(f'No .gguf file matching "{suffix}" — available: {candidates}')
        raise MatricxonError(f'Ambiguous suffix "{suffix}", matches multiple files — available: {candidates}')

    async def _download(self, repo_id: str, matched: dict, dest: Path, partial_path: Path) -> AsyncIterator[dict]:
        dest.parent.mkdir(parents=True, exist_ok=True)
        total = matched["size_bytes"]
        digest = hashlib.sha256()
        completed = 0
        last_yield = 0

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
                url = f"https://huggingface.co/{repo_id}/resolve/main/{matched['filename']}"
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    with partial_path.open("wb") as f:
                        async for chunk in response.aiter_bytes(_CHUNK_SIZE):
                            f.write(chunk)
                            digest.update(chunk)
                            completed += len(chunk)
                            if completed - last_yield >= _PROGRESS_THROTTLE_BYTES:
                                last_yield = completed
                                yield {
                                    "status": f"pulling {matched['filename']}",
                                    "completed": completed,
                                    "total": total,
                                }
        except httpx.HTTPError as exc:
            partial_path.unlink(missing_ok=True)
            raise MatricxonError(f"download of {matched['filename']!r} failed: {self._describe(exc)}") from exc
        except BaseException:
            # Covers cancellation too (asyncio.CancelledError is a BaseException, not an Exception) — a
            # client disconnecting mid-download (see this module's own docstring on the real bug that
            # motivated a matching fix on Matricxon's own side) must not leave a half-written .partial file
            # behind either.
            partial_path.unlink(missing_ok=True)
            raise

        if matched["sha256"] is not None and digest.hexdigest() != matched["sha256"]:
            partial_path.unlink(missing_ok=True)
            raise MatricxonError(f"downloaded file failed sha256 verification: {matched['filename']!r}")

        partial_path.rename(dest)
        yield {"status": f"pulling {matched['filename']}", "completed": completed, "total": total}

    @staticmethod
    async def _probe(repo_id: str, filename: str) -> tuple[str | None, int | None]:
        probed = await gguf_probe.probe_metadata(repo_id, filename, proxy_url=None)
        if probed is None:
            return None, None
        return probed["architecture"], probed["parameter_count"]

    @staticmethod
    async def _infer_capabilities(repo_id: str, filename: str, architecture: str | None) -> list[str]:
        """Calls Matricxon's own `/api/infer-capabilities` (../matricxon/app/routers/capabilities_router.py,
        added 2026-09-21 specifically for this) rather than reimplementing `CapabilityInferer`'s own logic
        here — a small, no-download, no-filesystem call that stays correct even if Matricxon's own rules
        change later, instead of a second copy that would silently drift. `architecture=None` (the real GGUF
        header probe couldn't determine it) skips the call entirely — an unknown architecture can never have
        real capabilities either, same as CATALOG entries' own "architecture unknown" handling elsewhere."""
        if architecture is None:
            return []
        host = matricxon_pool.pick_host()
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
            try:
                resp = await client.post(
                    f"{host}/api/infer-capabilities",
                    json={"repo_id": repo_id, "filename": filename, "architecture": architecture},
                )
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise MatricxonError(
                    f"Could not reach Matricxon at {host}: {MatricxonDirectPuller._describe(exc)}"
                ) from exc
        return resp.json()["capabilities"]

    @staticmethod
    def _format_parameter_size(parameter_count: int | None) -> str:
        """Mirrors ../matricxon/app/pull/job.py's own `_format_parameter_size` — a tiny (5-line), purely
        cosmetic display-string formatter, duplicated rather than exposed over HTTP since it's not real
        domain logic (unlike CapabilityInferer above) and calling out to Matricxon for it would be needless
        round-trip overhead for something this simple."""
        if parameter_count is None:
            return "unknown"
        if parameter_count >= 1e9:
            return f"{parameter_count / 1e9:.1f}B"
        return f"{parameter_count / 1e6:.0f}M"

    def _write_sidecar(
        self,
        gguf_path: Path,
        tag: str,
        architecture: str | None,
        parameter_count: int | None,
        capabilities: list[str],
        context_length: int | None,
    ) -> None:
        """Same `InstalledModel` shape (../matricxon/app/models/installed_model.py) Matricxon's own
        `PullJob._write_sidecar` writes, plain `json.dumps` with no sort/indent to match — Matricxon
        discovers installed models only via this exact `*.gguf.json` sidecar file next to each `.gguf`
        (../matricxon/app/models/catalog.py's `ModelCatalog.list_installed`), never by scanning for bare
        `.gguf` files, so an inaccurate or missing sidecar means the model stays invisible to it."""
        architecture = architecture or "unknown"
        installed = {
            "tag": tag,
            "path": str(gguf_path),
            "architecture": architecture,
            "capabilities": capabilities,
            "size_bytes": gguf_path.stat().st_size,
            "family": architecture,
            "parameter_size": self._format_parameter_size(parameter_count),
            "context_length": context_length or 0,
        }
        sidecar_path = gguf_path.parent / f"{gguf_path.stem}.gguf.json"
        sidecar_path.write_text(json.dumps(installed))
