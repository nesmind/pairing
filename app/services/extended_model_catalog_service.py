"""
Admin-managed "browse more models" catalog for Settings > Model (see
app/model_catalog.py for the small, hand-curated default list shown up
front unchanged — this is the separate, admin-extensible list behind
"Browse more models"). Stored as one AppSetting JSON row (system-wide,
not per-user — like app.services.model_catalog_service.HiddenModelTags),
since this is catalog curation, not a personal preference.

Sourced entirely from Hugging Face (see
app.services.huggingface_client.HuggingFaceCatalogSearch) — search/verify
never touches Ollama's own registry. The model still gets *pulled*
through Ollama (its existing hf.co/ passthrough — see add's own
docstring on how the tag is built), this module just never uses Ollama
to find or confirm one exists.
"""

from __future__ import annotations

import re

from sqlalchemy.ext.asyncio import AsyncSession

from app import hardware
from app.model_catalog import CuratedModel
from app.models import SYSTEM_OWNER_ID, AppSetting, User
from app.schemas import CatalogEntry
from app.services.extended_model_catalog_enrichment import HuggingFaceModelProbe
from app.services.huggingface_client import HuggingFaceCatalogSearch, HuggingFaceLookupError
from app.services.inference_client import list_models
from app.services.matricxon_support_checker import MatricxonSupportChecker
from app.services.model_catalog_service import HiddenModelTags


class ExtendedModelCatalog:
    """The one place admin-added Hugging Face entries are stored, verified, and turned into catalog rows —
    the extended-catalog analogue of app.services.model_catalog_service.ChatModelCatalogBuilder."""

    _KEY = "extended_model_catalog"

    ALREADY_INSTALLED_NOTE = "Already installed on this system — added without an internet lookup."

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    @staticmethod
    def build_tag(repo_id: str, filename: str) -> str:
        """`hf.co/<repo_id>:<filename minus its .gguf extension>` — see HuggingFaceCatalogSearch's own docstring
        on why the full filename (not a hand-extracted quant code) is used."""
        stem = filename.removesuffix(".gguf")
        return f"hf.co/{repo_id}:{stem}"

    @staticmethod
    def _repo_display_name(repo_id: str) -> str:
        name = repo_id.rstrip("/").rsplit("/", 1)[-1]
        return re.sub(r"[-_]?gguf$", "", name, flags=re.IGNORECASE) or name

    @staticmethod
    def _min_ram_gb(download_gb: float | None) -> float:
        """Ollama's own rule-of-thumb multiplier — same OLLAMA_RAM_ESTIMATE_MULTIPLIER
        app.services.model_catalog_service uses for auto-discovered installed models, duplicated here as a
        literal since importing it back would create a circular import (model_catalog_service imports nothing
        from this module, and shouldn't start to just for one constant)."""
        return round(download_gb * 1.25, 1) if download_gb else 0.0

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
        """Verifies `filename` still exists in `repo_id` on Hugging Face (never trusts a caller's earlier
        search/repo_files result), probes its real GGUF header for architecture/quantizations, and stores it.
        Raises ValueError if the tag is already in the catalog (only checked when it isn't already installed —
        see below) or the file is no longer offered.

        An already-installed tag still gets the same real lookup (confirmed live, 2026-09-22: skipping it
        unconditionally left the entry with a generic engine-reported family like "Llama" forever, instead of
        the real repo-derived name like "Llama-3.2-3B-Instruct" — a genuinely worse result than doing the same
        lookup every other tag gets), but never lets a failed one block registering something that's already on
        disk: a HuggingFaceLookupError (network down, repo gone, rate-limited) falls back to the old
        no-internet-lookup shape below instead of propagating, so silencing the "not in catalog" badge for an
        already-installed model never requires connectivity."""
        tag = self.build_tag(repo_id, filename)
        entries = await self.list()
        already_installed = tag in {m["name"] for m in await list_models()}
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

        entries = [e for e in entries if e["tag"] != tag]
        entries.append(entry)
        await self._save(entries)
        return entry

    async def _probe_entry(self, repo_id: str, filename: str, proxy_url: str | None) -> dict | None:
        """The real Hugging Face lookup + GGUF header probe behind add() — returns None if `filename` is no
        longer listed in `repo_id` (the caller decides whether that's a hard error or an acceptable fallback).
        `note` isn't set here — add() sets it based on install status, which this method has no view of."""
        tag = self.build_tag(repo_id, filename)
        repo = await HuggingFaceCatalogSearch.repo_files(repo_id, proxy_url)
        matched = next((f for f in repo["files"] if f["filename"] == filename), None)
        if matched is None:
            return None
        probed = await HuggingFaceModelProbe.probe(repo_id, filename, proxy_url)
        architecture = probed["architecture"]
        is_projector = architecture == "clip" if architecture else matched.get("is_projector", False)
        display_name = self._repo_display_name(repo_id)
        family = f"{display_name} (vision projector)" if is_projector else display_name
        parameter_size = None if is_projector else repo["parameter_size"]
        return {
            "tag": tag,
            "family": family,
            "parameter_size": parameter_size,
            "download_gb": matched["download_gb"],
            "is_projector": is_projector,
            # Same "repo also ships an mmproj sidecar" signal the HF file picker's own +Vision badge uses (see
            # HuggingFaceCatalogSearch.repo_files) — Ollama auto-pairs it on pull, Matricxon pulls it alongside.
            "vision": not is_projector and any(f.get("is_projector", False) for f in repo["files"]),
            "architecture": architecture,
            "quantizations": probed["quantizations"],
            "note": None,
        }

    async def remove(self, tag: str) -> None:
        entries = [e for e in await self.list() if e["tag"] != tag]
        await self._save(entries)

    async def build(self, user: User) -> list[CatalogEntry]:
        """Every admin-added entry not yet installed and not hidden from `user` — the extended-catalog analogue
        of ChatModelCatalogBuilder.build."""
        installed_models = await list_models()
        installed_tags = {m["name"] for m in installed_models}

        is_admin = user.role == "admin"
        hidden_tags = await HiddenModelTags(self._db).get()
        checker = await MatricxonSupportChecker.load(installed_models)
        capacity_gb = hardware.available_capacity_gb()

        result = []
        for entry in await self.list():
            tag = entry["tag"]
            if tag in hidden_tags and not is_admin:
                continue
            if tag in installed_tags:
                continue
            min_ram_gb = self._min_ram_gb(entry.get("download_gb"))
            is_projector = entry.get("is_projector", False)
            verdict = checker.verdict_for(
                entry.get("architecture"),
                entry.get("quantizations"),
                is_projector=is_projector,
            )
            result.append(
                CatalogEntry(
                    family=entry.get("family") or "Other",
                    vendor="Other",
                    tag=tag,
                    parameter_size=entry.get("parameter_size") or "?",
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
                    matricxon_supported=verdict.supported,
                    matricxon_unsupported_reason=verdict.reason,
                    is_projector=is_projector,
                    is_auto_discovered=True,
                )
            )
        return result

    async def find(self, tag: str) -> CuratedModel | None:
        """Same shape app.model_catalog.CuratedCatalog.find returns, so a caller resolving a tag across both the
        curated catalog and this admin-added one (see app/routers/settings.py's pull_model) can treat them
        identically."""
        entry = next((e for e in await self.list() if e["tag"] == tag), None)
        if entry is None:
            return None
        return CuratedModel(
            family=entry.get("family") or "Other",
            vendor="Other",
            tag=tag,
            parameter_size=entry.get("parameter_size") or "?",
            context_length=None,
            download_gb=entry.get("download_gb"),
            min_ram_gb=self._min_ram_gb(entry.get("download_gb")),
            locally_runnable=True,
            architecture=entry.get("architecture") or "unknown",
            quantizations=entry.get("quantizations"),
        )
