"""Which models exist and which a user may see/pick: the curated catalog (app/model_catalog.py) cross-referenced
with what's installed, the admin hidden-tags list and the default for new accounts.

installed_chat_models/installed_vision_models/installed_embedding_models/resolve_installed_model and the six
default-model get/set functions stay module-level functions (delegating to
default_model_settings.DefaultModelSettings where relevant): several unrelated features import them
(chat/conversation/channel services, document ingest/retrieval, user_service, routers/users.py, and
tests/conftest.py's fixture), and reshaping them would ripple widely. HiddenModelTags/ChatModelCatalogBuilder,
used only here and in routers/settings.py, are classes.
"""

import re

from sqlalchemy.ext.asyncio import AsyncSession

from app import hardware
from app.model_catalog import CATALOG
from app.models import SYSTEM_OWNER_ID, AppSetting, User
from app.schemas import CatalogEntry, ModelCatalogResponse
from app.services import settings_service
from app.services.default_model_settings import DefaultModelSettings
from app.services.engine_support_checker import OLLAMA_RAM_ESTIMATE_MULTIPLIER, EngineSupportSet
from app.services.engines.registry import registry
from app.services.extended_model_catalog_enrichment import HuggingFaceModelProbe
from app.services.inference_client import list_models
from app.services.installed_projector_catalog import InstalledProjectorCatalog

# Matches the quant token at the end of a tag suffix ("Q3_K_M" from "...:Llama-3.2-3B-Instruct-Q3_K_M", "Q8_0"
# from "...:Q8_0"), a parameter_size display fallback (_auto_discovered_entry). Not
# extended_model_catalog_enrichment.HuggingFaceModelProbe.guess_quantizations_from_filename: that collapses
# "Q3_K_M"/"Q3_K_S"/"Q3_K_L" to the coarse "Q3_K" for Matricxon matching; this wants the exact token, more
# informative than "?" though it answers a different question.
_QUANT_TOKEN_RE = re.compile(r"(?:IQ\d_[A-Z]+|Q\d(?:_\d)?(?:_K)?(?:_[SML])?|BF16|F16|FP16|F32|FP32)$", re.IGNORECASE)


async def installed_chat_models() -> list[str]:
    """Tags of pulled models that can hold a conversation (the "completion" filter ChatModelCatalogBuilder uses)."""
    installed = await list_models()
    return [m["name"] for m in installed if "completion" in m.get("capabilities", [])]


async def installed_vision_models() -> list[str]:
    """Tags of installed models that accept images ("vision"), for the "default vision model" picker
    (get_default_vision_model)."""
    installed = await list_models()
    return [m["name"] for m in installed if "vision" in m.get("capabilities", [])]


async def installed_embedding_models() -> list[str]:
    """Tags of installed models that can embed text ("embedding"); the analogue of installed_vision_models."""
    installed = await list_models()
    return [m["name"] for m in installed if "embedding" in m.get("capabilities", [])]


async def get_default_model_for_new_users(db: AsyncSession, installed: list[str]) -> str | None:
    return await DefaultModelSettings(db).for_new_users(installed)


async def set_default_model_for_new_users(db: AsyncSession, model: str) -> None:
    await DefaultModelSettings(db).set_for_new_users(model)


async def resolve_installed_model(db: AsyncSession, user: User, model: str) -> str:
    """If `model` isn't installed on the active engine, falls back to one that is: `user`'s default, then the
    system default for new users, then the first installed. Returns `model` unchanged if installed or if no
    fallback exists (the caller's attempt then fails clearly; this never raises).

    Needed because a conversation's `model` doesn't change on an engine switch (engine_service) and Ollama and
    Matricxon have separate catalogs; sending after a switch failed with a raw "404 Not Found", reading as if the
    switch hadn't worked. Callers persist a changed result (chat_service.build_reply_stream), so it runs once per
    conversation.
    """
    installed = await installed_chat_models()
    if model in installed:
        return model
    user_default = await settings_service.get_default_model(db, user)
    if user_default in installed:
        return user_default
    return await DefaultModelSettings(db).for_new_users(installed) or model


async def get_default_vision_model(db: AsyncSession) -> str | None:
    """The admin-configured model for messages with an image attachment (see DefaultModelSettings.vision)."""
    return await DefaultModelSettings(db).vision()


async def set_default_vision_model(db: AsyncSession, model: str) -> None:
    await DefaultModelSettings(db).set_vision(model)


async def get_default_embedding_model(db: AsyncSession) -> str | None:
    """The admin-configured model inference_client.embed uses for the Knowledge base (see
    DefaultModelSettings.embedding)."""
    return await DefaultModelSettings(db).embedding()


async def set_default_embedding_model(db: AsyncSession, model: str) -> None:
    await DefaultModelSettings(db).set_embedding(model)


class HiddenModelTags:
    """Tags an admin hid from regular users' picker; system-wide. Used only here and in
    extended_model_catalog_service."""

    _KEY = "hidden_model_tags"

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def get(self) -> set[str]:
        row = await self._db.get(AppSetting, (SYSTEM_OWNER_ID, self._KEY))
        return set(row.value["tags"]) if row else set()

    async def blocks(self, user: User, model: str) -> bool:
        """True when `model` is disabled and `user` isn't an admin (admins see and use every model)."""
        return user.role != "admin" and model in await self.get()

    async def set(self, tags: set[str]) -> None:
        row = await self._db.get(AppSetting, (SYSTEM_OWNER_ID, self._KEY))
        value = {"tags": sorted(tags)}
        if row is None:
            self._db.add(AppSetting(owner_id=SYSTEM_OWNER_ID, key=self._KEY, value=value))
        else:
            row.value = value
        await self._db.commit()


class ChatModelCatalogBuilder:
    """Every model Settings' Model tab offers: the curated list (app/model_catalog.py), chat-capable models already
    pulled but not in it (e.g. pulled on the terminal) and installed vision-projector sidecars with no
    "completion" capability (InstalledProjectorCatalog), so the picker never hides something owned. Each entry
    says whether it's installed and whether the hardware (app/hardware.py) can run it. Embedding-only models are
    excluded.

    A model an admin hid (HiddenModelTags) is omitted entirely for regular users (not just marked client-side),
    so hidden tags can't be found in the API response. Admins get every entry, flagged via `hidden`, to unhide.
    """

    def __init__(self, db: AsyncSession, user: User) -> None:
        self._db = db
        self._user = user
        self._hidden_tags = HiddenModelTags(db)
        self._installed_by_name: dict[str, dict] = {}

    async def installed_tags(self) -> list[str]:
        """Installed chat-capable tags this user may switch a conversation to (the chat page's switcher). Admin-hidden
        tags are left out for regular users, as in build; a hidden model an existing conversation uses is unaffected.
        """
        installed = await installed_chat_models()
        if self._user.role == "admin":
            return installed
        hidden = await self._hidden_tags.get()
        return [m for m in installed if m not in hidden]

    async def build(self) -> ModelCatalogResponse:
        installed_models = await list_models()
        is_admin = self._user.role == "admin"
        hidden_tags = await self._hidden_tags.get()
        installed_by_tag = {m["name"]: m for m in installed_models if "completion" in m.get("capabilities", [])}
        self._installed_by_name = {m["name"]: m for m in installed_models}
        capacity_gb = hardware.available_capacity_gb()
        support_set = await EngineSupportSet.load(installed_models)

        entries = []
        for entry in CATALOG.chat_models:
            if entry.tag in hidden_tags and not is_admin:
                continue
            entries.append(self._catalog_entry(entry, installed_by_tag, hidden_tags, capacity_gb, support_set))

        catalog_tags = {entry.tag for entry in CATALOG.chat_models if entry.tag}
        extended_catalog_entries = await self._extended_catalog_entries()
        for name, installed_model in installed_by_tag.items():
            if name in catalog_tags or (name in hidden_tags and not is_admin):
                continue
            extended_entry = extended_catalog_entries.get(name)
            if extended_entry is not None and is_admin and "discovered_via_engine" not in extended_entry:
                # Pre- data, registered before this field existed and never engine-scoped. `name` is confirmed
                # installed under the active engine here (it's in installed_by_tag): the one chance to backfill before
                # an
                # uninstall makes it unrecoverable (backfill_discovered_via_engine).
                from app.services.extended_model_catalog_service import ExtendedModelCatalog

                await ExtendedModelCatalog(self._db).backfill_discovered_via_engine(name, registry.active().name)
                extended_entry["discovered_via_engine"] = registry.active().name
            entries.append(self._auto_discovered_entry(name, installed_model, hidden_tags, support_set, extended_entry))

        entries.extend(
            InstalledProjectorCatalog.entries(installed_models, catalog_tags, hidden_tags, is_admin, support_set)
        )
        return ModelCatalogResponse(
            entries=entries,
            hardware=hardware.hardware_summary(),
            model_management=registry.active().capabilities.model_management,
        )

    async def _extended_catalog_entries(self) -> dict[str, dict]:
        """Tag -> entry for everything the admin registered via "Browse more models". An installed one must not get the
        is_auto_discovered=True "Not in catalog" badge of a stray install (it read as an error), and its
        HF-sourced family/parameter_size (ExtendedModelCatalog.add) beats Matricxon's often "unknown" report (the
        same model showed "Llama unknown"). Local import: extended_model_catalog_service imports HiddenModelTags
        from here, so a top-level import would be circular.
        """
        from app.services.extended_model_catalog_service import ExtendedModelCatalog

        return {entry["tag"]: entry for entry in await ExtendedModelCatalog(self._db).list()}

    @staticmethod
    def _vendor_from_tag(tag: str) -> str:
        """A `hf.co/<org>/<repo>:<suffix>` tag encodes its publishing org; a hardcoded "Other" showed vendor "Other"
        instead of "unsloth". "Other" only for tags with no org (e.g. a plain Ollama-library tag pulled on
        the terminal).
        """
        if not tag.startswith("hf.co/"):
            return "Other"
        repo_id = tag.removeprefix("hf.co/").split(":", 1)[0]
        return repo_id.split("/", 1)[0] or "Other"

    @staticmethod
    def _clean_engine_reported(value: str | None) -> str | None:
        """Matricxon's /api/tags reports the literal "unknown" for fields it doesn't know (every installed model's
        `details.parameter_size`,; curated entries' static value wins first). It's a truthy string, so
        `value or "?"` never caught it; treated as missing so it falls through to "?".
        """
        if value and value.strip().lower() != "unknown":
            return value
        return None

    @staticmethod
    def _quant_from_tag(tag: str) -> str | None:
        """Best-effort exact quantization label from the tag suffix (see _QUANT_TOKEN_RE; alongside, not instead of,
        the coarser Matricxon-compatibility guess). None if no recognizable token (e.g. a plain Ollama-library tag),
        so callers fall through like _clean_engine_reported.
        """
        suffix = tag.rsplit(":", 1)[-1] if ":" in tag else tag
        match = _QUANT_TOKEN_RE.search(suffix)
        return match.group(0).upper() if match else None

    def _catalog_entry(
        self, entry, installed_by_tag: dict, hidden_tags: set[str], capacity_gb: float, support_set: EngineSupportSet
    ):
        tag = entry.tag
        # A projector has no "completion" capability, so it's never in installed_by_tag.
        lookup = self._installed_by_name if entry.is_projector else installed_by_tag
        installed_model = lookup.get(tag) if tag else None
        installed = installed_model is not None
        # Once installed, the engine's live report is the only source of architecture, with no merge or fallback to
        # the curated guess (a pre-install estimate; same as `vision` below). Trusting it when the engine disagrees or
        # reports nothing would let a stale value outrank what runs.
        architecture = (
            self._clean_engine_reported(installed_model.get("details", {}).get("family"))
            if installed_model is not None
            else entry.architecture
        )
        engine_support = support_set.support_for(
            tag=tag,
            architecture=architecture,
            quantizations=entry.quantizations,
            is_projector=entry.is_projector,
            download_gb=entry.download_gb,
            min_ram_gb=entry.min_ram_gb,
        )
        return CatalogEntry(
            family=entry.family,
            vendor=entry.vendor,
            tag=tag,
            parameter_size=entry.parameter_size,
            context_length=entry.context_length,
            download_gb=entry.download_gb,
            min_ram_gb=entry.min_ram_gb,
            engine_support=engine_support,
            locally_runnable=entry.locally_runnable,
            installed=installed,
            # Installed models skip the live check: they ran well enough to download, and re-blocking them is worse than
            # trusting that.
            hardware_ok=installed or (entry.locally_runnable and capacity_gb >= entry.min_ram_gb),
            unavailable_reason=entry.unavailable_reason,
            hidden=tag in hidden_tags,
            # The curated flag is a pre-install promise (moondream's is true because Ollama auto-pairs its mmproj on an
            # hf.co/ pull); Matricxon doesn't pair, so trusting it post-install could show "+vision" while
            # installed_vision_models (the live check _auto_discovered_entry uses) says no. Once installed the
            # engine's
            # capabilities are the truth; the static flag is a fallback for tags not .
            vision=(
                "vision" in installed_model.get("capabilities", []) if installed_model is not None else entry.vision
            ),
            # Every CATALOG entry is a normal chat model (curated as one); see CatalogEntry.text_capable.
            text_capable=not entry.is_projector,
            is_projector=entry.is_projector,
            # No pre-install guess exists (CatalogEntry.chat_format_unverified): read live only; False (default) until
            # installed.
            chat_format_unverified=(
                "chat_format_unverified" in installed_model.get("capabilities", [])
                if installed_model is not None
                else False
            ),
        )

    def _auto_discovered_entry(
        self,
        name: str,
        installed_model: dict,
        hidden_tags: set[str],
        support_set: EngineSupportSet,
        extended_entry: dict | None,
    ) -> CatalogEntry:
        details = installed_model.get("details", {})
        size_gb = (installed_model.get("size") or 0) / 1_000_000_000
        ollama_ram_gb = round(size_gb * OLLAMA_RAM_ESTIMATE_MULTIPLIER, 1) if size_gb else 0
        # Prefer ExtendedModelCatalog's HF-sourced family/parameter_size (_extended_catalog_entries) over Matricxon's
        # per-tag report, often the literal "unknown": truthy, it skipped the `or` fallback and rendered "Llama unknown"
        # . _clean_engine_reported filters it case-insensitively (applied to family defensively; not yet
        # observed).
        # A connector engine (e.g. cloudflare_client) names its models itself: its tags are no Hugging Face paths.
        display_name = installed_model.get("display_name")
        family = (
            display_name
            or (extended_entry or {}).get("family")
            or (self._clean_engine_reported(details.get("family")) or "Other").title()
        )
        if display_name:  # already says what the model is; a tag-based guess would repeat it
            parameter_size = self._clean_engine_reported(details.get("parameter_size")) or ""
        else:
            parameter_size = (
                (extended_entry or {}).get("parameter_size")
                or self._clean_engine_reported(details.get("parameter_size"))
                or self._quant_from_tag(name)
                or "?"
            )
        # Bug: this assumed "installed while Matricxon is active" meant "proven to run there".
        # Matricxon's
        # /api/tags lists anything with a valid sidecar (written at pull time by its puller or matricxon_direct_puller),
        # whether or not it ever loaded; architecture support is checked at first chat, so a model could look fine then
        # fail with "unsupported architecture". Architecture prefers ExtendedModelCatalog's GGUF-probed value for a
        # registered
        # tag, else what the engine reports: both /api/tags report the GGUF general.architecture as details.family
        # whichever
        # engine is active, so even an Ollama-installed model gets a real verdict, not the old unconditional "not
        # verified".
        architecture = (extended_entry or {}).get("architecture") or self._clean_engine_reported(details.get("family"))
        quantizations = (extended_entry or {}).get(
            "quantizations"
        ) or HuggingFaceModelProbe.guess_quantizations_from_filename(name)
        # support_set.support_for is the single check every other entry (curated, extended, embedding) uses; sharing it
        # avoids a second guess that could disagree (the bug above). download_gb is passed raw so each engine's RAM
        # formula gives the same figure as ollama_ram_gb above, not a second rounding.
        engine_support = support_set.support_for(
            tag=name,
            architecture=architecture,
            quantizations=quantizations,
            download_gb=size_gb or None,
            min_ram_gb=ollama_ram_gb,
        )
        return CatalogEntry(
            family=family,
            vendor=installed_model.get("vendor") or self._vendor_from_tag(name),
            tag=name,
            parameter_size=parameter_size,
            context_length=details.get("context_length"),
            download_gb=round(size_gb, 1) if size_gb else None,
            min_ram_gb=ollama_ram_gb,
            engine_support=engine_support,
            locally_runnable=True,
            installed=True,
            hardware_ok=True,
            unavailable_reason=None,
            hidden=name in hidden_tags,
            # Already installed, so both read the engine's capability tags, not a curated guess like the CATALOG branch
            # above.
            vision="vision" in installed_model.get("capabilities", []),
            text_capable="completion" in installed_model.get("capabilities", []),
            chat_format_unverified="chat_format_unverified" in installed_model.get("capabilities", []),
            is_auto_discovered=extended_entry is None,
        )
