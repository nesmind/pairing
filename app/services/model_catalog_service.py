"""
Everything about which models exist and which ones a given user is allowed to see/pick: the curated
catalog (app/model_catalog.py) cross-referenced with what's actually installed, the admin-configurable
hidden-tags list, and the admin-configured default for brand-new accounts.

installed_chat_models/installed_vision_models/installed_embedding_models/resolve_installed_model and the six
default-model get/set functions stay plain module-level functions (delegating to
app.services.default_model_settings.DefaultModelSettings where relevant) rather than becoming class methods —
they're imported directly by several unrelated features (chat/conversation/channel services, document
ingest/retrieval, user_service, app/routers/users.py) and tests/conftest.py's own shared fixture; reshaping them
would ripple far outside this catalog-building feature for no real benefit. HiddenModelTags/
ChatModelCatalogBuilder below, with no callers outside this module and app/routers/settings.py, are genuinely
converted to classes.
"""

import re

from sqlalchemy.ext.asyncio import AsyncSession

from app import hardware
from app.model_catalog import CATALOG
from app.models import SYSTEM_OWNER_ID, AppSetting, User
from app.schemas import CatalogEntry, ModelCatalogResponse
from app.services import settings_service
from app.services.default_model_settings import DefaultModelSettings
from app.services.extended_model_catalog_enrichment import HuggingFaceModelProbe
from app.services.inference_client import list_models
from app.services.installed_projector_catalog import InstalledProjectorCatalog
from app.services.matricxon_support_checker import MatricxonSupportChecker

# Ollama's own API exposes no per-model RAM estimate at all (unlike Matricxon's real `estimated_ram_gb`, see
# MatricxonSupportChecker) - this is the same rule-of-thumb headroom-above-download-size multiplier this file
# already used, before CatalogEntry split its RAM figure per engine, for auto-discovered installed models not
# in the curated CATALOG. Named and reused here instead of two independent copies of the same magic number.
OLLAMA_RAM_ESTIMATE_MULTIPLIER = 1.25

# Matches the exact quant token as written at the end of a tag's own suffix (e.g. "Q3_K_M" out of
# "...:Llama-3.2-3B-Instruct-Q3_K_M", "Q8_0" out of "...:Q8_0") — used as a fallback for parameter_size display
# when nothing else is known (see _auto_discovered_entry). Deliberately not app.services.
# extended_model_catalog_enrichment.HuggingFaceModelProbe.guess_quantizations_from_filename: that one
# collapses "Q3_K_M"/"Q3_K_S"/"Q3_K_L" down to the same coarse "Q3_K" family for Matricxon-compatibility
# matching — this wants the exact token as written, which is more informative than a bare "?" even though it
# answers a different question (quantization, not parameter count).
_QUANT_TOKEN_RE = re.compile(r"(?:IQ\d_[A-Z]+|Q\d(?:_\d)?(?:_K)?(?:_[SML])?|BF16|F16|FP16|F32|FP32)$", re.IGNORECASE)


async def installed_chat_models() -> list[str]:
    """Tags of every model actually pulled that can hold a conversation — same "completion" capability filter
    ChatModelCatalogBuilder uses for its own picker."""
    installed = await list_models()
    return [m["name"] for m in installed if "completion" in m.get("capabilities", [])]


async def installed_vision_models() -> list[str]:
    """Tags of every installed model that can accept an image ("vision" capability) — used for the Settings >
    System "default vision model" picker (see get_default_vision_model below)."""
    installed = await list_models()
    return [m["name"] for m in installed if "vision" in m.get("capabilities", [])]


async def installed_embedding_models() -> list[str]:
    """Tags of every installed model that can embed text ("embedding" capability) — the embedding analogue of
    installed_vision_models above."""
    installed = await list_models()
    return [m["name"] for m in installed if "embedding" in m.get("capabilities", [])]


async def get_default_model_for_new_users(db: AsyncSession, installed: list[str]) -> str | None:
    return await DefaultModelSettings(db).for_new_users(installed)


async def set_default_model_for_new_users(db: AsyncSession, model: str) -> None:
    await DefaultModelSettings(db).set_for_new_users(model)


async def resolve_installed_model(db: AsyncSession, user: User, model: str) -> str:
    """If `model` isn't actually installed on the currently active engine, falls back to *some* model that is —
    `user`'s own default, then the system-wide default for new users, then whichever installed model comes
    first. Returns `model` unchanged if it's already installed, or if nothing can serve as a fallback either
    (the caller's own attempt then fails with a clear error the normal way — this never raises on its own).

    Exists for the real gap this surfaced: a conversation's own `model` column never changes on its own when an
    admin switches the active engine (see app.services.engine_service) — Ollama and Matricxon each have their
    own separate catalog, so a tag installed on one is routinely not installed on the other. Without this,
    sending a message after an engine switch failed with a raw "404 Not Found," confusing enough that it read
    as the *switch itself* not having worked. Callers persist a changed result back (see
    chat_service.build_reply_stream), so this only ever needs to run once per conversation until corrected."""
    installed = await installed_chat_models()
    if model in installed:
        return model
    user_default = await settings_service.get_default_model(db, user)
    if user_default in installed:
        return user_default
    return await DefaultModelSettings(db).for_new_users(installed) or model


async def get_default_vision_model(db: AsyncSession) -> str | None:
    """The admin-configured model any message with an image attachment is answered by — see
    app.services.default_model_settings.DefaultModelSettings.vision for the real fallback logic."""
    return await DefaultModelSettings(db).vision()


async def set_default_vision_model(db: AsyncSession, model: str) -> None:
    await DefaultModelSettings(db).set_vision(model)


async def get_default_embedding_model(db: AsyncSession) -> str | None:
    """The admin-configured model app.services.inference_client.embed calls use for the Knowledge base (RAG)
    feature — see app.services.default_model_settings.DefaultModelSettings.embedding for the real fallback
    logic."""
    return await DefaultModelSettings(db).embedding()


async def set_default_embedding_model(db: AsyncSession, model: str) -> None:
    await DefaultModelSettings(db).set_embedding(model)


class HiddenModelTags:
    """Tags an admin has hidden from regular users' model picker — system-wide, not per-user. No callers
    outside this module and app.services.extended_model_catalog_service."""

    _KEY = "hidden_model_tags"

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def get(self) -> set[str]:
        row = await self._db.get(AppSetting, (SYSTEM_OWNER_ID, self._KEY))
        return set(row.value["tags"]) if row else set()

    async def set(self, tags: set[str]) -> None:
        row = await self._db.get(AppSetting, (SYSTEM_OWNER_ID, self._KEY))
        value = {"tags": sorted(tags)}
        if row is None:
            self._db.add(AppSetting(owner_id=SYSTEM_OWNER_ID, key=self._KEY, value=value))
        else:
            row.value = value
        await self._db.commit()


class ChatModelCatalogBuilder:
    """Every model Settings' Model tab can offer: the curated list in app/model_catalog.py, plus anything
    chat-capable already pulled into the ML engine that isn't in that list (e.g. pulled manually on the terminal) or
    is an installed vision-projector sidecar with no "completion" capability of its own (see
    InstalledProjectorCatalog), so the picker never hides something the user already has. Each entry says
    whether it's installed and whether this machine's hardware (app/hardware.py) can actually run it —
    embedding-only models are excluded, same as before, since they can't hold a conversation.

    A model an admin has hidden (see HiddenModelTags) is left out of the response entirely for a regular user —
    not just marked hidden client-side — so there's no way to discover a hidden tag by inspecting the API
    response. An admin gets every entry, including hidden ones (flagged via `hidden`), so they can unhide."""

    def __init__(self, db: AsyncSession, user: User) -> None:
        self._db = db
        self._user = user
        self._hidden_tags = HiddenModelTags(db)

    async def installed_tags(self) -> list[str]:
        """Installed chat-capable model tags this user is allowed to switch a conversation to — used by the
        chat page's model-switch picker. Admin-hidden tags are left out for a regular user, same visibility
        rule as build(); an already-hidden model an existing conversation is using isn't affected by that."""
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
        capacity_gb = hardware.available_capacity_gb()
        checker = await MatricxonSupportChecker.load(installed_models)

        entries = []
        for entry in CATALOG.chat_models:
            if entry.tag in hidden_tags and not is_admin:
                continue
            entries.append(self._catalog_entry(entry, installed_by_tag, hidden_tags, capacity_gb, checker))

        catalog_tags = {entry.tag for entry in CATALOG.chat_models if entry.tag}
        extended_catalog_entries = await self._extended_catalog_entries()
        for name, installed_model in installed_by_tag.items():
            if name in catalog_tags or (name in hidden_tags and not is_admin):
                continue
            extended_entry = extended_catalog_entries.get(name)
            if extended_entry is None and is_admin:
                extended_entry = await self._self_register(name)
            entries.append(self._auto_discovered_entry(name, installed_model, hidden_tags, checker, extended_entry))

        entries.extend(
            InstalledProjectorCatalog.entries(installed_models, catalog_tags, hidden_tags, is_admin, checker)
        )
        return ModelCatalogResponse(entries=entries, hardware=hardware.hardware_summary())

    async def _extended_catalog_entries(self) -> dict[str, dict]:
        """Tag -> entry for everything the admin deliberately registered via "Browse more models". An
        installed one of these must not get the same is_auto_discovered=True "Not in catalog" badge as a
        genuinely stray/unregistered install (confirmed live, 2026-09-22: an admin-added-then-pulled model
        showed it, reading as if something had gone wrong) — and its real, HF-sourced family/parameter_size
        (see ExtendedModelCatalog.add) is worth preferring over Matricxon's own often-"unknown" raw report for
        the same reason (confirmed live the same day: the same model showed as "Llama unknown"). Local import:
        extended_model_catalog_service already imports HiddenModelTags from this module, so a top-level import
        here would be circular; deferred until this actually runs."""
        from app.services.extended_model_catalog_service import ExtendedModelCatalog

        return {entry["tag"]: entry for entry in await ExtendedModelCatalog(self._db).list()}

    async def _self_register(self, tag: str) -> dict | None:
        """Auto-heals the exact gap _extended_catalog_entries' own docstring describes, instead of leaving an
        installed-but-never-added model stuck showing a generic engine-reported name and the "Not in catalog"
        badge until an admin happens to notice and manually re-add it (confirmed live, 2026-09-22: the same
        handful of models kept resurfacing this exact bug across engine switches and new pulls, each needing a
        one-off manual fix). Runs on every admin's own catalog view (see build's is_admin gate — a regular
        user's view never writes to the admin-curated catalog) for any not-yet-registered hf.co/ tag; a plain
        Ollama-library tag (no "hf.co/" prefix) is skipped outright since there's no Hugging Face repo to look
        up at all. Best-effort: any failure (network down, repo gone, rate-limited, an unexpected response
        shape) is swallowed so a catalog page load never breaks over this — the tag still renders with today's
        existing fallback and gets another chance to self-heal on the next load."""
        if not tag.startswith("hf.co/") or ":" not in tag:
            return None
        repo_id, _, suffix = tag.removeprefix("hf.co/").partition(":")
        from app.services.extended_model_catalog_service import ExtendedModelCatalog

        try:
            return await ExtendedModelCatalog(self._db).add(repo_id, f"{suffix}.gguf", proxy_url=None)
        except Exception:  # noqa: BLE001 - best-effort self-heal, see this method's own docstring
            return None

    @staticmethod
    def _vendor_from_tag(tag: str) -> str:
        """A real `hf.co/<org>/<repo>:<suffix>` tag always encodes its real publishing org — confirmed live,
        2026-09-22: an admin-added-then-pulled model showed vendor "Other" (hardcoded regardless of tag shape)
        instead of its real org ("unsloth"). Falls back to "Other" only for a tag with no real org to read
        (e.g. a plain Ollama-library tag pulled manually on the terminal)."""
        if not tag.startswith("hf.co/"):
            return "Other"
        repo_id = tag.removeprefix("hf.co/").split(":", 1)[0]
        return repo_id.split("/", 1)[0] or "Other"

    @staticmethod
    def _clean_engine_reported(value: str | None) -> str | None:
        """Matricxon's own /api/tags reports the literal string "unknown" for a field it genuinely doesn't
        know (confirmed live, 2026-09-22: every installed model's own `details.parameter_size` — even a
        curated one, though those never reach this fallback since CATALOG's own static value takes
        precedence) — that sentinel is a real, truthy string, so a plain `value or "?"` fallback never
        catches it. Treated the same as missing data here so it can fall through to a real "?" instead."""
        if value and value.strip().lower() != "unknown":
            return value
        return None

    @staticmethod
    def _quant_from_tag(tag: str) -> str | None:
        """Best-effort exact quantization label pulled straight from the tag's own suffix — see
        _QUANT_TOKEN_RE's own module-level comment for why this exists alongside, not instead of, the coarser
        Matricxon-compatibility guess. None if the suffix doesn't end in a recognizable quant token at all
        (a plain Ollama-library tag with no hf.co/ suffix, say) — same "let a caller fall through further"
        contract as _clean_engine_reported."""
        suffix = tag.rsplit(":", 1)[-1] if ":" in tag else tag
        match = _QUANT_TOKEN_RE.search(suffix)
        return match.group(0).upper() if match else None

    def _catalog_entry(self, entry, installed_by_tag: dict, hidden_tags: set[str], capacity_gb: float, checker):
        tag = entry.tag
        installed_model = installed_by_tag.get(tag) if tag else None
        installed = installed_model is not None
        # Once a tag is actually installed, the active engine's own live report is the only source of truth for
        # architecture — not a merge/fallback with the curated static guess. A curated entry's architecture is
        # only ever a pre-install estimate (see this method's own vision comment below for the exact same
        # principle already applied there); trusting it once the engine disagrees, or even as a silent fallback
        # when the engine reports nothing, would let a stale/wrong static value outrank what's actually running.
        architecture = (
            self._clean_engine_reported(installed_model.get("details", {}).get("family"))
            if installed_model is not None
            else entry.architecture
        )
        verdict = checker.verdict_for(architecture, entry.quantizations)
        # Ollama's own RAM figure from download_gb (it exposes no better number); Matricxon's real figure below
        # only once this exact tag is confirmed installed (an un-pulled tag's can't be computed without its
        # real GGUF file to read tensor shapes from).
        min_ram_gb_ollama = (
            round(entry.download_gb * OLLAMA_RAM_ESTIMATE_MULTIPLIER, 1) if entry.download_gb else entry.min_ram_gb
        )
        return CatalogEntry(
            family=entry.family,
            vendor=entry.vendor,
            tag=tag,
            parameter_size=entry.parameter_size,
            context_length=entry.context_length,
            download_gb=entry.download_gb,
            min_ram_gb=entry.min_ram_gb,
            min_ram_gb_ollama=min_ram_gb_ollama,
            min_ram_gb_matricxon=checker.estimated_ram_gb(tag) if tag else None,
            locally_runnable=entry.locally_runnable,
            installed=installed,
            # Already-installed models get a pass on the live check — they clearly ran well enough to be
            # downloaded before, and re-blocking something the user already has would be a worse experience
            # than trusting past success.
            hardware_ok=installed or (entry.locally_runnable and capacity_gb >= entry.min_ram_gb),
            unavailable_reason=entry.unavailable_reason,
            hidden=tag in hidden_tags,
            # The curated flag is only ever a pre-install promise (e.g. moondream's is true because Ollama
            # auto-pairs its mmproj projector on a hf.co/ pull — confirmed live, 2026-09-22) — Matricxon doesn't
            # do that same auto-pairing, so trusting it post-install could show a "+vision" badge here while
            # installed_vision_models() (the same live-capability check _auto_discovered_entry uses below)
            # correctly reports the model as not vision-capable. Once a tag is actually installed, the engine's
            # own reported capabilities are the real source of truth; the static flag is only a fallback for a
            # tag nothing has confirmed live yet.
            vision=(
                "vision" in installed_model.get("capabilities", []) if installed_model is not None else entry.vision
            ),
            # Every CATALOG entry is a normal chat model by definition (this whole list is curated as one) —
            # see CatalogEntry.text_capable's own docstring for why this is never actually False here today.
            text_capable=True,
            matricxon_supported=verdict.supported,
            matricxon_unsupported_reason=verdict.reason,
        )

    def _auto_discovered_entry(
        self,
        name: str,
        installed_model: dict,
        hidden_tags: set[str],
        checker: MatricxonSupportChecker,
        extended_entry: dict | None,
    ) -> CatalogEntry:
        details = installed_model.get("details", {})
        size_gb = (installed_model.get("size") or 0) / 1_000_000_000
        ollama_ram_gb = round(size_gb * OLLAMA_RAM_ESTIMATE_MULTIPLIER, 1) if size_gb else 0
        # Prefer ExtendedModelCatalog's own real, HF-sourced family/parameter_size (see _extended_catalog_
        # entries' own docstring) over Matricxon's raw per-tag report, which is frequently just the literal
        # string "unknown" — confirmed live, 2026-09-22, sailing straight through the `or` fallback below (a
        # truthy string, so it never fell through to "?") and rendering as the literal display "Llama unknown".
        # _clean_engine_reported filters that sentinel out (case-insensitively — same reasoning applies to
        # family, defensively, even though it hasn't been observed reporting "unknown" itself).
        family = (extended_entry or {}).get("family") or (
            self._clean_engine_reported(details.get("family")) or "Other"
        ).title()
        parameter_size = (
            (extended_entry or {}).get("parameter_size")
            or self._clean_engine_reported(details.get("parameter_size"))
            or self._quant_from_tag(name)
            or "?"
        )
        # Real bug found live, 2026-09-22: this used to assume "installed while Matricxon is the active
        # engine" meant "proven to run there" — false. Matricxon's own /api/tags lists anything with a valid
        # sidecar file (written at pull time, by either its own puller or app.services.matricxon_direct_puller)
        # regardless of whether it's ever actually been loaded — real architecture support is only checked at
        # load time, the first time something tries to chat with it. A model could show up here looking fine,
        # then fail with a genuine "unsupported architecture" error from Matricxon on the very first message.
        # checker.verdict_for is the one real check every other catalog entry (curated, extended, embedding)
        # already goes through — the single source of truth this now shares too, instead of a second,
        # independent guess that could (and did) disagree with it. Architecture prefers ExtendedModelCatalog's
        # own real GGUF-probed value when this tag is registered, else whichever engine reported it installed —
        # both Ollama's and Matricxon's own /api/tags report the real GGUF general.architecture string as
        # details.family regardless of which engine is currently active, so this gives a meaningful verdict
        # even for an Ollama-installed model, not the unconditional "not verified" the old code gave every
        # install that wasn't currently running on Matricxon.
        architecture = (extended_entry or {}).get("architecture") or self._clean_engine_reported(details.get("family"))
        quantizations = (extended_entry or {}).get(
            "quantizations"
        ) or HuggingFaceModelProbe.guess_quantizations_from_filename(name)
        verdict = checker.verdict_for(architecture, quantizations)
        return CatalogEntry(
            family=family,
            vendor=self._vendor_from_tag(name),
            tag=name,
            parameter_size=parameter_size,
            context_length=details.get("context_length"),
            download_gb=round(size_gb, 1) if size_gb else None,
            min_ram_gb=ollama_ram_gb,
            min_ram_gb_ollama=ollama_ram_gb,
            min_ram_gb_matricxon=checker.estimated_ram_gb(name),
            locally_runnable=True,
            installed=True,
            hardware_ok=True,
            unavailable_reason=None,
            hidden=name in hidden_tags,
            # Already installed, so both of these read Ollama's own real capability tags for it, not a
            # hand-curated guess like the CATALOG branch above.
            vision="vision" in installed_model.get("capabilities", []),
            text_capable="completion" in installed_model.get("capabilities", []),
            matricxon_supported=verdict.supported,
            matricxon_unsupported_reason=verdict.reason,
            is_auto_discovered=extended_entry is None,
        )
