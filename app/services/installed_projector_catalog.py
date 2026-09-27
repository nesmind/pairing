"""
Installed vision-projector (mmproj) entries for Settings > Model's main list — split out of
app.services.model_catalog_service (file-size rule), and a distinct concern from that file's own
CATALOG/auto-discovered-chat-model loops: a projector file has no "completion" capability of its own (never a
standalone chat model), so it used to be excluded from every list entirely once installed — confirmed live
(2026-09-21): that left an admin with no way to confirm a pull actually succeeded, or to manage/uninstall a
stray one, since it was invisible everywhere.
"""

import re

from app.schemas import CatalogEntry, EngineModelSupport
from app.services.engine_support_checker import EngineSupportSet


class InstalledProjectorCatalog:
    @staticmethod
    def _display_name(tag: str) -> str:
        """ "hf.co/concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf:llama-joycaption-...-f16" ->
        "llama-joycaption-beta-one-hf-llava-mmproj" — same "the repo's own name, not raw GGUF metadata"
        preference app.services.extended_model_catalog_service._repo_display_name already established (confirmed
        live that trusting in-file metadata for *display* can be actively unhelpful), adapted here for a full
        "hf.co/<repo>:<file>" tag instead of a bare repo_id."""
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
        """Every installed tag reporting a "clip" (vision-projector) architecture and no chat capability of its
        own — `installed_models` is the caller's own already-fetched list_models() result, so this makes no
        extra network call of its own. Ollama users should never actually see any of these: unlike Matricxon,
        Ollama bundles a same-repo mmproj into its main model's own manifest instead of listing it as an
        independent tag — this isn't Ollama-specific code, it just naturally has nothing to match there."""
        entries = []
        for model in installed_models:
            name = model["name"]
            if name in catalog_tags or (name in hidden_tags and not is_admin):
                continue
            if model.get("details", {}).get("family") != "clip":
                continue
            size_gb = (model.get("size") or 0) / 1_000_000_000
            # Not support_set.support_for: every engine's own real verdict_for(is_projector=True) call would
            # say "not supported" (a projector is never a standalone chat model) — genuinely true, but the
            # frontend never actually shows the "Not supported" badge for an is_projector entry regardless (see
            # renderModelRow's own is_projector check), so showing it as unsupported would just be misleading
            # noise. Each engine's own real RAM estimate is still worth showing, so that part is read directly.
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
                    family=f"{InstalledProjectorCatalog._display_name(name)} (vision projector)",
                    vendor="Other",
                    tag=name,
                    parameter_size="?",
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
