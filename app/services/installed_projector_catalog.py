"""Installed vision-projector (mmproj) entries for Settings > Model's main list, split from model_catalog_service
(file-size rule). Projectors have no "completion" capability, so they were excluded from every list once
installed, leaving admins unable to confirm a pull or uninstall a stray one.
"""

import re

from app.schemas import CatalogEntry, EngineModelSupport
from app.services.engine_support_checker import EngineSupportSet


class InstalledProjectorCatalog:
    @staticmethod
    def _vendor_from_tag(tag: str) -> str:
        """Same as `ChatModelCatalogBuilder._vendor_from_tag`, copied because that module imports this one (circular);
        see its docstring. A hardcoded "Other" put a projector under a different vendor heading than its paired chat
        model from the same repo; reading the projector's own tag fixes it without the paired tag.
        """
        if not tag.startswith("hf.co/"):
            return "Other"
        repo_id = tag.removeprefix("hf.co/").split(":", 1)[0]
        return repo_id.split("/", 1)[0] or "Other"

    @staticmethod
    def _display_name(tag: str) -> str:
        """ "hf.co/concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf:llama-joycaption-...-f16" ->
        "llama-joycaption-beta-one-hf-llava-mmproj": the repo's name, not raw GGUF metadata (as
        extended_model_catalog_service._repo_display_name; in-file metadata can mislead), for a full
        "hf.co/<repo>:<file>" tag.
        """
        repo_id = tag.removeprefix("hf.co/").split(":", 1)[0]
        name = repo_id.rsplit("/", 1)[-1]
        return re.sub(r"[-_]?gguf$", "", name, flags=re.IGNORECASE) or name

    @staticmethod
    def entries(
        installed_models: list[dict],
        catalog_tags: set[str],
        hidden_tags: set[str],
        is_admin: bool,
        support_set: EngineSupportSet,
    ) -> list[CatalogEntry]:
        """Installed tags reporting a "clip" (vision-projector) architecture and no chat capability.
        `installed_models` is the caller's list_models result (no extra call). Ollama shows none: it bundles a
        same-repo mmproj into the main model's manifest, so nothing matches (no Ollama-specific code).
        """
        entries = []
        for model in installed_models:
            name = model["name"]
            if name in catalog_tags or (name in hidden_tags and not is_admin):
                continue
            if model.get("details", {}).get("family") != "clip":
                continue
            size_gb = (model.get("size") or 0) / 1_000_000_000
            # Not support_set.support_for: verdict_for(is_projector=True) says "not supported" (true, a projector is
            # never a chat model), but the frontend never shows that badge for is_projector (renderModelRow), so it
            # would
            # be noise. Each engine's RAM estimate is still read directly.
            engine_support = {
                engine_name: EngineModelSupport(
                    supported=True,
                    reason=None,
                    min_ram_gb=support_set.checker_for(engine_name).estimated_ram_gb(
                        name, download_gb=round(size_gb, 1) if size_gb else None
                    ),
                )
                for engine_name in support_set.engine_names()
            }
            entries.append(
                CatalogEntry(
                    # "(vision projector)" is now a "+Vision"/"Vision" badge (renderModelRow's is_projector in
                    # settings.js),
                    # not text in this string.
                    family=InstalledProjectorCatalog._display_name(name),
                    vendor=InstalledProjectorCatalog._vendor_from_tag(name),
                    tag=name,
                    # A vision tower + projector has no parameter count, and "?" appeared as a stray " ?" in the row
                    # title
                    # (modelName joins family + parameter_size); empty so the join's `.filter(Boolean)` drops it.
                    parameter_size="",
                    download_gb=round(size_gb, 1) if size_gb else None,
                    min_ram_gb=0,
                    engine_support=engine_support,
                    locally_runnable=True,
                    installed=True,
                    hardware_ok=True,
                    hidden=name in hidden_tags,
                    vision=False,
                    text_capable=False,
                    is_projector=True,
                    is_auto_discovered=True,
                )
            )
        return entries
