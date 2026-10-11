"""Admin-managed "browse more models" catalog for Settings > Model (the small curated default list is
app/model_catalog.py). Stored as one system-wide AppSetting JSON row (catalog curation, not a personal
preference; like model_catalog_service.HiddenModelTags).

Sourced from Hugging Face (huggingface_client.HuggingFaceCatalogSearch). Models are still *pulled* through
Ollama's hf.co/ passthrough (see add for the tag); Ollama is never used to find or confirm one.
"""

from __future__ import annotations

import re

from sqlalchemy.ext.asyncio import AsyncSession

from app import hardware
from app.model_catalog import CuratedModel
from app.models import SYSTEM_OWNER_ID, AppSetting, User
from app.schemas import CatalogEntry
from app.services.engine_support_checker import OLLAMA_RAM_ESTIMATE_MULTIPLIER, EngineSupportSet
from app.services.engines.registry import registry
from app.services.extended_model_catalog_enrichment import HuggingFaceModelProbe
from app.services.huggingface_client import HuggingFaceCatalogSearch, HuggingFaceLookupError
from app.services.inference_client import list_models
from app.services.model_catalog_service import HiddenModelTags


class ExtendedModelCatalog:
    """Stores, verifies and turns admin-added Hugging Face entries into catalog rows; the analogue of
    ChatModelCatalogBuilder.
    """

    _KEY = "extended_model_catalog"

    ALREADY_INSTALLED_NOTE = "Already installed on this system — added without an internet lookup."

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    @staticmethod
    def build_tag(repo_id: str, filename: str) -> str:
        """`hf.co/<repo_id>:<filename minus.gguf>` (the full filename, not a quant code; see
        HuggingFaceCatalogSearch)."""
        stem = filename.removesuffix(".gguf")
        return f"hf.co/{repo_id}:{stem}"

    @staticmethod
    def _repo_display_name(repo_id: str) -> str:
        name = repo_id.rstrip("/").rsplit("/", 1)[-1]
        return re.sub(r"[-_]?gguf$", "", name, flags=re.IGNORECASE) or name

    @staticmethod
    def _vendor_from_tag(tag: str) -> str:
        """Like ModelCatalogService._vendor_from_tag, duplicated to avoid a circular import. Tags are always
        `hf.co/<org>/<repo>:<suffix>` (build_tag), so the org is real; it used to return "Other" always, so two
        "unsloth" entries rendered under separate vendor headings.
        """
        if not tag.startswith("hf.co/"):
            return "Other"
        repo_id = tag.removeprefix("hf.co/").split(":", 1)[0]
        return repo_id.split("/", 1)[0] or "Other"

    @staticmethod
    def _min_ram_gb(download_gb: float | None) -> float:
        """Ollama's rule-of-thumb multiplier (OLLAMA_RAM_ESTIMATE_MULTIPLIER, as OllamaSupportChecker)."""
        return round(download_gb * OLLAMA_RAM_ESTIMATE_MULTIPLIER, 1) if download_gb else 0.0

    async def list(self) -> list[dict]:
        row = await self._db.get(AppSetting, (SYSTEM_OWNER_ID, self._KEY))
        return list(row.value["entries"]) if row else []

    async def _save(self, entries: list[dict]) -> None:
        row = await self._db.get(AppSetting, (SYSTEM_OWNER_ID, self._KEY))
        value = {"entries": entries}
        if row is None:
            self._db.add(AppSetting(owner_id=SYSTEM_OWNER_ID, key=self._KEY, value=value))
        else:
            row.value = value
        await self._db.commit()

    async def add(self, repo_id: str, filename: str, proxy_url: str | None) -> dict:
        """Verifies `filename` still exists in `repo_id` on Hugging Face (never trusting an earlier search), probes
        its GGUF header for architecture/quantizations, and stores it. Raises ValueError if the tag is already
        catalogued (checked only when not installed) or the file is no longer offered.

        An installed tag gets the same lookup (skipping it left a generic family like "Llama" forever instead of
        "Llama-3.2-3B-Instruct"), but a HuggingFaceLookupError (network down, repo gone, rate limit)
        falls back to the no-lookup shape below, so silencing "not in catalog" never needs connectivity.

        Every entry records `discovered_via_engine` (the active engine), including admin searches and adds
        (found in two rounds): the list is engine-specific like the installed-models list, so an entry
        found under Ollama must not resurface as a suggestion when Matricxon becomes active, whether it was
        noticed on disk or searched by hand.
        """
        from app.services.image_model_service import ImageModelStore  # lazy: it imports this module

        tag = self.build_tag(repo_id, filename)
        entries = await self.list()
        already_installed = tag in {m["name"] for m in await list_models()} or (
            await ImageModelStore.open(self._db)
        ).is_installed(tag)
        if not already_installed and any(e["tag"] == tag for e in entries):
            raise ValueError(f'"{tag}" is already in the catalog.')

        try:
            entry = await self._probe_entry(repo_id, filename, proxy_url)
        except HuggingFaceLookupError:
            if not already_installed:
                raise
            entry = None
        if entry is None:
            if not already_installed:
                raise ValueError(f'"{filename}" is no longer available in {repo_id} on Hugging Face.')
            entry = {"tag": tag, "family": None, "parameter_size": None, "download_gb": None}
        if already_installed:
            entry["note"] = self.ALREADY_INSTALLED_NOTE
        entry["discovered_via_engine"] = registry.active().name

        entries = [e for e in entries if e["tag"] != tag]
        entries.append(entry)
        await self._save(entries)
        return entry

    async def _probe_entry(self, repo_id: str, filename: str, proxy_url: str | None) -> dict | None:
        """The Hugging Face lookup + GGUF header probe behind add; None if `filename` is no longer in `repo_id`
        (the caller decides if that's an error). `note` is set by add from install status.
        """
        tag = self.build_tag(repo_id, filename)
        repo = await HuggingFaceCatalogSearch.repo_files(repo_id, proxy_url)
        matched = next((f for f in repo["files"] if f["filename"] == filename), None)
        if matched is None:
            return None
        if repo.get("is_image"):
            # A diffusion model: no chat-engine GGUF header; it installs into the image engine's store
            # (ImageModelStore), whatever chat engine is active.
            return {
                "tag": tag,
                "kind": "image",
                "family": self._repo_display_name(repo_id),
                "parameter_size": None,
                "download_gb": matched["download_gb"],
                "note": None,
            }
        probed = await HuggingFaceModelProbe.probe(repo_id, filename, proxy_url)
        architecture = probed["architecture"]
        is_projector = architecture == "clip" if architecture else matched.get("is_projector", False)
        # "(vision projector)" is now a badge (settings.js renderModelRow is_projector, shared with
        # installed_projector_catalog.py), so this is just the plain display name.
        display_name = self._repo_display_name(repo_id)
        family = display_name
        parameter_size = None if is_projector else repo["parameter_size"]
        return {
            "tag": tag,
            "family": family,
            "parameter_size": parameter_size,
            "download_gb": matched["download_gb"],
            "is_projector": is_projector,
            # Same "repo ships an mmproj sidecar" signal as the HF file picker's +Vision badge
            # (HuggingFaceCatalogSearch.repo_files); Ollama auto-pairs it on pull, Matricxon pulls it alongside.
            "vision": not is_projector and any(f.get("is_projector", False) for f in repo["files"]),
            "architecture": architecture,
            "quantizations": probed["quantizations"],
            "note": None,
        }

    async def remove(self, tag: str) -> None:
        entries = [e for e in await self.list() if e["tag"] != tag]
        await self._save(entries)

    async def clear(self) -> None:
        """Empties the "Browse more models" suggestions on any engine change, mode switch or restart. Auto-registered
        entries (ALREADY_INSTALLED_NOTE) stay: never shown as suggestions (see build), but they cache an installed
        model's name/size, and dropping them made the next Model tab load re-query Hugging Face (~30s).
        """
        # Image-model entries are engine-independent, so they survive too.
        kept = [
            e for e in await self.list() if e.get("note") == self.ALREADY_INSTALLED_NOTE or e.get("kind") == "image"
        ]
        await self._save(kept)

    async def backfill_discovered_via_engine(self, tag: str, engine_name: str) -> None:
        """Stamps `discovered_via_engine` onto an entry registered before: while `tag` is confirmed
        installed under `engine_name` (ChatModelCatalogBuilder.build's auto-discovered loop): the only moment the
        engine is known; it can't be recovered once uninstalled. Checked as key *absence* (new entries always set
        a real name). No-op once backfilled.
        """
        entries = await self.list()
        changed = False
        for entry in entries:
            if entry["tag"] == tag and "discovered_via_engine" not in entry:
                entry["discovered_via_engine"] = engine_name
                changed = True
        if changed:
            await self._save(entries)

    async def build(self, user: User) -> list[CatalogEntry]:
        """Admin-added entries not installed under the active engine, relevant to it and not hidden from `user`; the
        engine-specific analogue of ChatModelCatalogBuilder.build.

        Closes a gap (two rounds): entries record the engine active when added (see add), so a tag
        found installed under Matricxon, or added by hand while Ollama was active, doesn't resurface as a new
        "Browse more models" suggestion under another engine, even once uninstalled everywhere: the recorded engine
        is the only remaining answer to "is this new to this engine" (see backfill_discovered_via_engine).
        """
        installed_models = await list_models()
        installed_tags = {m["name"] for m in installed_models}
        active_engine_name = registry.active().name

        is_admin = user.role == "admin"
        hidden_tags = await HiddenModelTags(self._db).get()
        support_set = await EngineSupportSet.load(installed_models)
        capacity_gb = hardware.available_capacity_gb()

        result = []
        for entry in await self.list():
            tag = entry["tag"]
            if tag in hidden_tags and not is_admin:
                continue
            if entry.get("kind") == "image":
                continue  # shown in Settings > Model's Diffusion models section instead (ImageModelStore)
            if tag in installed_tags:
                continue
            if entry.get("note") == self.ALREADY_INSTALLED_NOTE:
                # Registered only because it was installed (auto-discovery); once uninstalled it's a leftover, not a
                # suggestion.
                continue
            discovered_via_engine = entry.get("discovered_via_engine")
            if discovered_via_engine is not None and discovered_via_engine != active_engine_name:
                continue
            min_ram_gb = self._min_ram_gb(entry.get("download_gb"))
            is_projector = entry.get("is_projector", False)
            engine_support = support_set.support_for(
                tag=tag,
                architecture=entry.get("architecture"),
                quantizations=entry.get("quantizations"),
                is_projector=is_projector,
                download_gb=entry.get("download_gb"),
                min_ram_gb=min_ram_gb,
            )
            result.append(
                CatalogEntry(
                    family=entry.get("family") or "Other",
                    vendor=self._vendor_from_tag(tag),
                    tag=tag,
                    # "?" for an unknown non-projector parameter count (a failed probe re-run, see add's fallback);
                    # projectors
                    # use a badge instead (see _probe_entry).
                    parameter_size="" if is_projector else entry.get("parameter_size") or "?",
                    context_length=None,
                    download_gb=entry.get("download_gb"),
                    min_ram_gb=min_ram_gb,
                    locally_runnable=True,
                    installed=False,
                    hardware_ok=capacity_gb >= min_ram_gb,
                    unavailable_reason=entry.get("note"),
                    hidden=tag in hidden_tags,
                    vision=entry.get("vision", False),
                    text_capable=True,
                    removable=is_admin,
                    engine_support=engine_support,
                    is_projector=is_projector,
                    is_auto_discovered=True,
                )
            )
        return result

    async def find(self, tag: str) -> CuratedModel | None:
        """Same shape as CuratedCatalog.find, so callers resolving a tag across both catalogs (routers/settings.py
        pull_model) treat them alike.
        """
        entry = next((e for e in await self.list() if e["tag"] == tag), None)
        if entry is None:
            return None
        return CuratedModel(
            family=entry.get("family") or "Other",
            vendor=self._vendor_from_tag(tag),
            tag=tag,
            parameter_size=entry.get("parameter_size") or "?",
            context_length=None,
            download_gb=entry.get("download_gb"),
            min_ram_gb=self._min_ram_gb(entry.get("download_gb")),
            locally_runnable=True,
            architecture=entry.get("architecture") or "unknown",
            quantizations=entry.get("quantizations"),
        )
