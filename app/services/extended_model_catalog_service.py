"""
Admin-managed "browse more models" catalog for Settings > Model (see
app/model_catalog.py for the small, hand-curated default list shown up
front unchanged — this is the separate, admin-extensible list behind
"Browse more models"). Stored as one AppSetting JSON row (system-wide,
not per-user — like app.services.model_catalog_service.get_hidden_tags),
since this is catalog curation, not a personal preference.

Sourced entirely from Hugging Face (see
app.services.huggingface_client) — search/verify never touches
Ollama's own registry. The model still gets *pulled* through Ollama
(its existing hf.co/ passthrough — see add_model's own docstring on
how the tag is built), this module just never uses Ollama to find or
confirm one exists.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app import hardware
from app.models import SYSTEM_OWNER_ID, AppSetting, User
from app.schemas import CatalogEntry
from app.services.huggingface_client import get_repo_files
from app.services.model_catalog_service import get_hidden_tags
from app.services.ollama_client import list_models

EXTENDED_CATALOG_KEY = "extended_model_catalog"

ALREADY_INSTALLED_NOTE = "Already installed on this system — added without an internet lookup."


def build_tag(repo_id: str, filename: str) -> str:
    """ "Qwen/Qwen2.5-3B-Instruct-GGUF" + "qwen2.5-3b-instruct-q4_k_m.gguf" ->
    "hf.co/Qwen/Qwen2.5-3B-Instruct-GGUF:qwen2.5-3b-instruct-q4_k_m" — Ollama's own hf.co/ pull syntax (already
    used by app/model_catalog.py's MiniMax entry), which matches the tag suffix against candidate filenames in
    the repo case-insensitively. The *full* filename (minus its extension) is used as the suffix rather than a
    hand-extracted "quant code" (e.g. just "Q4_K_M") — a full-filename match can never accidentally resolve to
    the wrong file the way a heuristic substring extraction could on an unusually-named repo."""
    stem = filename.removesuffix(".gguf")
    return f"hf.co/{repo_id}:{stem}"


async def get_extended_catalog(db: AsyncSession) -> list[dict]:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, EXTENDED_CATALOG_KEY))
    return list(row.value["entries"]) if row else []


async def _save_extended_catalog(db: AsyncSession, entries: list[dict]) -> None:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, EXTENDED_CATALOG_KEY))
    value = {"entries": entries}
    if row is None:
        db.add(AppSetting(owner_id=SYSTEM_OWNER_ID, key=EXTENDED_CATALOG_KEY, value=value))
    else:
        row.value = value
    await db.commit()


async def add_model(db: AsyncSession, repo_id: str, filename: str, proxy_url: str | None) -> dict:
    """Adds the GGUF file `filename` from Hugging Face repo `repo_id` to the extended catalog. Re-verifies
    against Hugging Face from scratch (see app.services.huggingface_client.get_repo_files) rather than trusting
    whatever the caller's own earlier search/file-list call returned, so the catalog only ever stores what the
    server itself just confirmed — *unless* the resulting tag is already installed (checked against Ollama's own
    /api/tags here), in which case the Hugging Face round trip is skipped entirely: the model obviously exists,
    so there's nothing left to verify, and this also means adding an already-installed model works even on a
    machine with no internet access. Raises ValueError if the tag is already in this catalog,
    HuggingFaceLookupError (from get_repo_files) on a genuinely unknown repo/file or a network/proxy problem —
    the caller (see app/routers/model_catalog_admin.py) turns either into a clean HTTP error."""
    tag = build_tag(repo_id, filename)
    entries = await get_extended_catalog(db)
    if any(e["tag"] == tag for e in entries):
        raise ValueError(f'"{tag}" is already in the catalog.')

    installed_tags = {m["name"] for m in await list_models()}
    if tag in installed_tags:
        entry = {
            "tag": tag,
            "family": None,
            "parameter_size": None,
            "download_gb": None,
            "note": ALREADY_INSTALLED_NOTE,
        }
    else:
        repo = await get_repo_files(repo_id, proxy_url)
        matched = next((f for f in repo["files"] if f["filename"] == filename), None)
        if matched is None:
            raise ValueError(f'"{filename}" is no longer available in {repo_id} on Hugging Face.')
        entry = {
            "tag": tag,
            "family": repo["family"],
            "parameter_size": repo["parameter_size"],
            "download_gb": matched["download_gb"],
            "note": None,
        }

    entries.append(entry)
    await _save_extended_catalog(db, entries)
    return entry


async def remove_model(db: AsyncSession, tag: str) -> None:
    entries = [e for e in await get_extended_catalog(db) if e["tag"] != tag]
    await _save_extended_catalog(db, entries)


def _min_ram_gb(download_gb: float | None) -> float:
    """Same "roughly 1.25x the on-disk weight size" heuristic app/model_catalog.py's own docstring documents for
    its hand-curated entries — applied here since an admin-added entry has no hand-picked min_ram_gb of its own,
    only whatever real download size the registry reported (or None, for an already-installed entry with no
    lookup at all — that one skips hardware gating entirely below, same as build_model_catalog's own
    already-installed branch)."""
    return round(download_gb * 1.25, 1) if download_gb else 0.0


async def build_extended_catalog(db: AsyncSession, user: User) -> list[CatalogEntry]:
    """The browsable response for GET /api/settings/model-catalog/extended — same CatalogEntry shape and
    installed/hidden/hardware_ok rules as app.services.model_catalog_service.build_model_catalog, just sourced
    from this admin-managed list instead of the static app/model_catalog.py one.

    An entry that's actually installed is left out of this response entirely, not just marked `installed` —
    build_model_catalog's own "installed but not in the static catalog" branch already surfaces it in the
    *default* list once pulled (same tag, same info, read live from Ollama there), so showing it here too would
    just be the same row duplicated across both lists. Still stored here regardless (see get_extended_catalog),
    so uninstalling it later makes it reappear in this view again with no re-verification needed."""
    installed_models = await list_models()
    installed_by_tag = {m["name"]: m for m in installed_models if "completion" in m.get("capabilities", [])}

    is_admin = user.role == "admin"
    hidden_tags = await get_hidden_tags(db)
    capacity_gb = hardware.available_capacity_gb()

    result = []
    for entry in await get_extended_catalog(db):
        tag = entry["tag"]
        if tag in hidden_tags and not is_admin:
            continue
        if tag in installed_by_tag:
            continue
        min_ram_gb = _min_ram_gb(entry.get("download_gb"))
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
                vision=False,
                text_capable=True,
                # Admin-only in the UI (see renderModelRow's own isAdmin check) — no `not installed` check
                # needed here: an installed entry never reaches this line at all (see the `continue` above).
                removable=is_admin,
            )
        )
    return result


async def find_entry(db: AsyncSession, tag: str) -> dict | None:
    """Same shape (locally_runnable/min_ram_gb/unavailable_reason keys) as app.model_catalog.find_entry, so
    app/routers/settings.py's pull-model endpoint can accept an extended-catalog tag the same way it already
    accepts a default-catalog one, without needing to know which list a given tag actually came from."""
    entry = next((e for e in await get_extended_catalog(db) if e["tag"] == tag), None)
    if entry is None:
        return None
    return {
        "tag": entry["tag"],
        "locally_runnable": True,
        "min_ram_gb": _min_ram_gb(entry.get("download_gb")),
        "unavailable_reason": None,
    }
