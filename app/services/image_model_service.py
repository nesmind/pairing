"""Diffusion models for the local stable-diffusion.cpp engine. An image model found through Settings > Model's
Hugging Face browser (see HuggingFaceCatalogSearch.is_image_repo) is downloaded straight from Hugging Face into
the engine's own models folder — never into Ollama/Matricxon, whose chat loaders reject it — and offered in the
engine form's Model dropdown. Only the *local* engine has a folder to manage; in Remote mode nothing happens here.

Where: the engine's "Models folder" if set, else a `diffusion` sibling of the Matricxon/Ollama models folder
(so everything lives under one tree: models/ollama, models/matricxon, models/diffusion), else the
engine install's own models folder.

Layout: <models dir>/<org>/<repo>/<file>.gguf, so a catalog tag (`hf.co/<org>/<repo>:<file minus .gguf>`) maps to
a path and back exactly. Loose files an admin dropped into the folder by hand are listed too (without a tag)."""

import logging
import re
from collections.abc import AsyncIterator
from pathlib import Path
from urllib.parse import quote

from sqlalchemy.ext.asyncio import AsyncSession

from app.model_catalog import CATALOG
from app.schemas import CatalogEntry, CompanionFile, ImageModelFile
from app.services import image_engine_service, sdcpp_installer, settings_service
from app.services.extended_model_catalog_service import ExtendedModelCatalog
from app.services.hf_download import stream_file
from app.services.image_companions import FOLDER as COMPANIONS_DIR, ImageCompanions

logger = logging.getLogger("llama_chat")

_REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")
# Rough working-set multiplier over the file size for a CPU diffusion run (weights + activations + VAE).
_RAM_MULTIPLIER = 1.3


async def resolve_models_dir(db: AsyncSession) -> Path:
    config = await image_engine_service.get_sdcpp_config(db)
    if config.models_path:
        return Path(config.models_path)
    for owner in (
        await settings_service.get_matricxon_server_config(db),
        await settings_service.get_ollama_server_config(db),
    ):
        if owner.models_path and Path(owner.models_path).name in ("matricxon", "ollama"):
            return Path(owner.models_path).parent / "diffusion"
    return sdcpp_installer.models_dir()


class ImageModelStore:
    EXTENSIONS = (".gguf", ".safetensors", ".ckpt")
    LOCAL_ONLY_MESSAGE = (
        "Image models install into the local stable-diffusion.cpp engine — switch it to Local mode "
        "(Settings > Image) first. A remote engine manages its own models."
    )

    def __init__(self, db: AsyncSession, root: Path) -> None:
        self._db = db
        self.root = root

    @classmethod
    async def open(cls, db: AsyncSession) -> "ImageModelStore":
        return cls(db, await resolve_models_dir(db))

    @staticmethod
    def parse_tag(tag: str) -> tuple[str, str]:
        """(repo_id, filename) for `hf.co/<org>/<repo>:<stem>`; ValueError for anything else, including path
        tricks — the tag is client-supplied and becomes a filesystem path."""
        if not tag.startswith("hf.co/") or ":" not in tag:
            raise ValueError("Not a Hugging Face model tag.")
        repo_id, stem = tag.removeprefix("hf.co/").split(":", 1)
        filename = f"{stem}.gguf"
        unsafe_parts = {"", ".", ".."}
        if (
            not _REPO_RE.match(repo_id)
            or unsafe_parts & set(repo_id.split("/"))
            or not stem
            or unsafe_parts & set(filename.split("/"))
        ):
            raise ValueError("Invalid model tag.")
        return repo_id, filename

    def path_for(self, tag: str) -> Path:
        repo_id, filename = self.parse_tag(tag)
        return self.root / repo_id / filename

    def tag_for(self, path: Path) -> str | None:
        """The catalog tag of a file that sits at <org>/<repo>/<file>.gguf, else None (a hand-placed file)."""
        parts = path.relative_to(self.root).parts
        if len(parts) < 3 or path.suffix != ".gguf":
            return None
        stem = "/".join(parts[2:]).removesuffix(".gguf")
        return f"hf.co/{parts[0]}/{parts[1]}:{stem}"

    def list_files(self) -> list[ImageModelFile]:
        root = self.root
        if not root.is_dir():
            return []
        files = [
            p for p in root.rglob("*") if p.is_file() and p.suffix in self.EXTENSIONS and COMPANIONS_DIR not in p.parts
        ]
        return [
            ImageModelFile(
                name=str(p.relative_to(root)),
                path=str(p),
                size_gb=round(p.stat().st_size / 1_000_000_000, 2),
                tag=self.tag_for(p),
            )
            for p in sorted(files)
        ]

    def is_installed(self, tag: str) -> bool:
        try:
            return self.path_for(tag).is_file()
        except ValueError:
            return False

    async def is_local(self) -> bool:
        return (await image_engine_service.get_sdcpp_config(self._db)).mode == "local"

    async def catalog_entries(self, is_admin: bool = False) -> list[CatalogEntry]:
        """Settings > Model's Diffusion models rows: the curated defaults, then admin-added image models, then any
        other tagged file already in the folder — each marked installed or not."""
        added = {e["tag"]: e for e in await ExtendedModelCatalog(self._db).list() if e.get("kind") == "image"}
        installed = {f.tag: f for f in self.list_files() if f.tag}
        curated = {m.tag: m for m in CATALOG.diffusion_models}
        entries = []
        for tag in [
            *curated,
            *(t for t in added if t not in curated),
            *(t for t in installed if t not in curated and t not in added),
        ]:
            file = installed.get(tag)
            repo_id = self.parse_tag(tag)[0]
            known, extra = curated.get(tag), added.get(tag, {})
            size_gb = file.size_gb if file else (known.download_gb if known else extra.get("download_gb"))
            entries.append(
                CatalogEntry(
                    family=(known.family if known else extra.get("family")) or repo_id.rsplit("/", 1)[-1],
                    vendor=known.vendor if known else repo_id.split("/", 1)[0],
                    tag=tag,
                    parameter_size=known.parameter_size if known else "",
                    download_gb=size_gb,
                    min_ram_gb=known.min_ram_gb if known else round((size_gb or 0) * _RAM_MULTIPLIER, 1),
                    locally_runnable=True,
                    installed=file is not None,
                    hardware_ok=True,
                    text_capable=False,
                    removable=is_admin and tag in added and file is None,
                    is_image=True,
                    companions=self.companions_for(tag),
                )
            )
        return entries

    async def prepare_pull(self, tag: str) -> tuple[str, str, Path]:
        """Validates a pull request and returns (repo_id, filename, destination). ValueError (-> 400) if the tag
        isn't a curated or admin-added image model, or the engine isn't local."""
        entries = await ExtendedModelCatalog(self._db).list()
        if CATALOG.find_diffusion(tag) is None and not any(
            e["tag"] == tag and e.get("kind") == "image" for e in entries
        ):
            raise ValueError("Unknown image model tag.")
        if not await self.is_local():
            raise ValueError(self.LOCAL_ONLY_MESSAGE)
        repo_id, filename = self.parse_tag(tag)
        return repo_id, filename, self.path_for(tag)

    async def download(self, repo_id: str, filename: str, dest: Path, proxy_url: str | None) -> AsyncIterator[dict]:
        """Streams the file from Hugging Face to `dest` (via a .part file, renamed once complete) as the same
        {"status","completed","total"} progress events a chat-model pull emits; the last event is either
        {"done": True} or {"error": ...}. On success, an unset engine Model is pointed at it."""
        url = f"https://huggingface.co/{repo_id}/resolve/main/{quote(filename)}"
        async for event in stream_file(url, dest, proxy_url, filename):
            yield event
            if "error" in event:
                return
        await self._default_model_if_unset(dest)
        yield {"done": True}

    async def download_companions(self, tag: str, flags: list[str], proxy_url: str | None) -> AsyncIterator[dict]:
        """Downloads only the companion files the user ticked (see image_companions); same event format."""
        async for event in ImageCompanions.download(self.path_for(tag), flags, proxy_url):
            yield event
        yield {"done": True}

    def companions_for(self, tag: str) -> list[CompanionFile]:
        path = self.path_for(tag)
        missing = {c.flag for c in ImageCompanions.missing(path)}
        return [
            CompanionFile(flag=c.flag, label=c.label, size_gb=c.size_gb, installed=c.flag not in missing)
            for c in ImageCompanions.for_model(path)
        ]

    async def _default_model_if_unset(self, path: Path) -> None:
        config = await image_engine_service.get_sdcpp_config(self._db)
        if not config.model_path:
            config.model_path = str(path)
            await image_engine_service.set_sdcpp_config(self._db, config)

    async def delete(self, tag: str) -> None:
        """Removes the file (and any now-empty org/repo folders) and clears the engine's Model if it was this one.
        ValueError for a bad tag or Remote mode; a missing file is fine (already gone)."""
        if not await self.is_local():
            raise ValueError(self.LOCAL_ONLY_MESSAGE)
        path = self.path_for(tag)
        path.unlink(missing_ok=True)
        if not any(  # companions are shared by every quant in this folder: keep them while another remains
            f.is_file() and f.suffix in self.EXTENSIONS for f in path.parent.iterdir()
        ):
            ImageCompanions.remove(path)
        for parent in path.parents:
            if parent == self.root or any(parent.iterdir()):
                break
            parent.rmdir()
        config = await image_engine_service.get_sdcpp_config(self._db)
        if config.model_path == str(path):
            config.model_path = None
            await image_engine_service.set_sdcpp_config(self._db, config)
