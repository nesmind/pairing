"""Downloads a Matricxon model directly from Hugging Face instead of proxying through Matricxon's `/api/pull`
(matricxon_admin's other path, still used for remote hosts), which was reported as "slowing it very much"
on a local install. It writes the `.gguf` and its sidecar into Matricxon's models directory
using the Hugging Face APIs of "Browse more models" (huggingface_client, gguf_probe). Matricxon takes no
part in the download, only one small `/api/infer-capabilities` call for `CapabilityInferer`, so its rules
aren't duplicated here and cannot drift.

Local Matricxon only (matricxon_admin decides): writing to disk needs a shared filesystem.

Accepted limitations:
  - The sidecar's `context_length` comes from the repo-level `gguf` block (HuggingFaceCatalogSearch.repo_files),
    not a per-file read like `architecture` (that block can misattribute architecture for non-primary files,
    see gguf_probe). Cosmetic; not worth extending the binary parser for a display field.
  - Settings > System's outbound proxy (http_proxy_service) isn't honored; the proxy-through-Matricxon path
    never did either (it used Matricxon's own network and proxy settings), so this is parity.
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
# Byte-based, not Matricxon's time-based throttle (../matricxon/app/pull/downloader.py): no other reason to
# import `time`, and a byte interval stops a progress line per 1MB chunk just as well.
_PROGRESS_THROTTLE_BYTES = 8 * 1024 * 1024


class MatricxonDirectPuller:
    async def pull_stream(self, tag: str) -> AsyncIterator[dict]:
        """Yields the NDJSON pull-progress of matricxon_admin.pull_model_stream (`status`, plus `completed`/`total`
        while downloading); routers/settings.py `event_stream` forwards it unchanged. Raises MatricxonError on any
        failure (as the proxy path), so inference_client's wrapping into InferenceError still works.
        """
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
        """`tag` is `hf.co/<repo_id>:<suffix>`. `suffix` is the full filename stem for tags this app builds
        (ExtendedModelCatalog.build_tag), but a curated default_models.json entry may use a short one (e.g. "Q8_0"
        for "all-MiniLM-L6-v2.Q8_0.gguf"); see _match.
        """
        if not tag.startswith("hf.co/") or ":" not in tag:
            raise MatricxonError(f'"{tag}" is not an hf.co/<repo>:<file> tag — cannot pull it directly.')
        repo_id, _, suffix = tag.removeprefix("hf.co/").partition(":")
        return repo_id, suffix

    @staticmethod
    async def _get_config() -> MatricxonServerConfig:
        """Fetches its own session: this chain (routers/settings.py pull_model -> inference_client.pull_model_stream ->
        matricxon_admin.pull_model_stream) has no request `db`, and threading one through the engine-agnostic
        dispatch for one engine's need isn't worth it (as reply_model_unload_service._persist_cancelled_reply).
        """
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
        """Re-fetches the real repo listing, never trusting a caller's size/sha256 (as ExtendedModelCatalog.add).
        Returns the matched file dict and the repo-level context_length (an approximation, see module docstring).
        """
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
        """httpx (and some stdlib) exceptions can stringify to "" (a transient failure showed "download of... failed:
        "). Falls back to the type name plus the request URL when httpx attached one.
        """
        message = str(exc)
        if message:
            return message
        request = getattr(exc, "request", None)
        if request is not None:
            return f"{type(exc).__name__} for {request.url}"
        return type(exc).__name__

    @staticmethod
    def _match(suffix: str, files: list[dict]) -> dict:
        """Exact-stem-then-substring matching like../matricxon/app/pull/hf_resolver.py HFRepoResolver._match, so a
        curated tag with a short suffix ("Q8_0" for "all-MiniLM-L6-v2.Q8_0.gguf") resolves, not just full
        stems.
        """
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
            # Covers cancellation (CancelledError is a BaseException): a client disconnecting mid-download must not
            # leave a
            # half-written.partial (a matching fix exists on Matricxon's side).
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
        """Calls Matricxon's `/api/infer-capabilities` (../matricxon/app/routers/capabilities_router.py)
        instead of copying `CapabilityInferer`, which could drift: a small call, no download or filesystem. An
        unknown `architecture` (header probe failed) skips it, since it can't have capabilities (as CATALOG entries).
        """
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
        """Mirrors../matricxon/app/pull/job.py `_format_parameter_size`: a 5-line cosmetic formatter, duplicated as it
        isn't domain logic (unlike CapabilityInferer) and an HTTP call would be needless overhead.
        """
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
        """Writes the `InstalledModel` shape (../matricxon/app/models/installed_model.py) as Matricxon's
        `PullJob._write_sidecar` does (plain `json.dumps`). Matricxon finds models only via each `*.gguf.json` sidecar
        (`ModelCatalog.list_installed` in../matricxon/app/models/catalog.py), never bare `.gguf` files, so a wrong or
        missing sidecar hides the model.
        """
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
