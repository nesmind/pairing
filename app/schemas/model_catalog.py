"""Request/response shapes for Settings > Model — the default catalog (app/model_catalog.py), the admin-managed
"browse more models" extended catalog (app/services/extended_model_catalog_service.py), and Hugging Face
search/lookup (app/services/huggingface_client.py). Split out of app/schemas/settings.py once this many
model-catalog-specific shapes pushed that file over CLAUDE.md's line cap."""

from pydantic import BaseModel, Field, computed_field

from app.schemas.common import EngineName
from app.schemas.settings import HardwareSummary


class EngineModelSupport(BaseModel):
    """One engine's own verdict on one model — see
    app.services.engine_support_checker.SupportVerdict/EngineSupportChecker.estimated_ram_gb, the source of
    truth this is built from (EngineSupportSet.support_for). Replaces the old per-engine-named fields
    (min_ram_gb_ollama/min_ram_gb_matricxon/matricxon_supported/matricxon_unsupported_reason below, kept only as
    computed properties for backward compatibility) with one dict keyed by EngineName, so a third engine needs
    no new field anywhere."""

    supported: bool = True
    reason: str | None = None
    # This engine's own real per-model RAM estimate, or None when unknown/not applicable (see
    # EngineSupportChecker.estimated_ram_gb's own docstring — never a guess extrapolated from a
    # *different* engine's own figure).
    min_ram_gb: float | None = None


class CatalogEntry(BaseModel):
    """One model Settings can offer — either already pulled into the ML engine,
    pullable-and-compatible, or blocked by hardware. See
    app/model_catalog.py for where these come from and
    app/services/settings_service.py for how `installed`/`hardware_ok`
    are computed."""

    family: str
    vendor: str
    tag: str | None
    parameter_size: str
    context_length: int | None = None
    download_gb: float | None = None
    # The hardware-gating threshold a not-yet-installed entry is checked against (see
    # ChatModelCatalogBuilder's hardware_ok / app.routers.settings's pull-time capacity check) — this is
    # app/model_catalog.py's own hand-typed figure, kept as-is for that purpose. For *display*, prefer
    # min_ram_gb_ollama/min_ram_gb_matricxon below — real, computed-per-engine numbers, not a
    # hand-typed guess that can drift from either engine's actual real requirement (confirmed live,
    # 2026-09-21: Matricxon dequantizes to bf16 before computing, so its real need for a real
    # Ministral-3B Q4_K_M file runs to ~7.5GB, not the ~3GB this field alone implied).
    min_ram_gb: float
    # Every registered engine's own verdict on this exact model (see EngineModelSupport above) — built by
    # app.services.engine_support_checker.EngineSupportSet.support_for, once per catalog build. The
    # min_ram_gb_ollama/min_ram_gb_matricxon/matricxon_supported/matricxon_unsupported_reason properties below
    # read from this and exist only so old callers/tests keep working unchanged; new code should read this
    # dict directly.
    engine_support: dict[EngineName, EngineModelSupport] = Field(default_factory=dict)
    locally_runnable: bool
    installed: bool
    hardware_ok: bool
    unavailable_reason: str | None = None
    # Whether this model can see an image attachment (see
    # app.services.chat_attachment_service and the "Default vision
    # model" picker in Settings). For a not-yet-installed curated
    # app/model_catalog.py entry, hand-annotated there as a pre-install
    # estimate only; once a tag is actually installed (curated or not),
    # read live from the active engine's own "vision" capability tag
    # instead — confirmed live, 2026-09-22: the curated flag reflects
    # Ollama's own auto-pairing behavior for a hf.co/ pull and can
    # disagree with what's actually installed under a different engine
    # (e.g. Matricxon, which doesn't auto-pair a projector the same way).
    vision: bool = False
    # Whether this model can also hold an ordinary text conversation —
    # true for every entry this catalog can currently produce (see
    # ChatModelCatalogBuilder's own "completion" capability filter, which
    # every entry here already passed), so `vision and text_capable`
    # ("+Vision", a normal chat model that also sees images, e.g.
    # gemma4:12b) is the only combination reachable today; `vision and
    # not text_capable` ("Vision" alone, a vision-only model with no
    # text chat ability) is modeled honestly for correctness but can't
    # actually appear in this catalog as currently scoped.
    text_capable: bool = True
    # Whether the active engine actually has *no real confirmation* of this model's true chat/
    # instruction format (Matricxon's own "chat_format_unverified" capability tag — see
    # ../matricxon/app/models/capabilities.py's own docstring: true for a "completion"-capable
    # model with no real chat_template and no mistral3 tokenizer, meaning Matricxon is falling
    # back to a best-effort guess). Only ever read live, post-install, like `vision` above — there
    # is no meaningful pre-install guess for this (unlike vision's HF-repo-tag hint, nothing in a
    # not-yet-downloaded repo's listing reveals whether its own instruction format will actually
    # be followed). Surfaced so the model list can flag "this might not behave like a normal chat
    # model" instead of looking identical to one that's fully confirmed — added 2026-09-27 after
    # Hebrew-Mistral-7B-Q5_K_M produced incoherent, non-chat-like output regardless of prompt
    # format, with nothing in the list distinguishing it from a well-behaved model.
    chat_format_unverified: bool = False
    # Whether an admin has hidden this model from regular users' picker
    # (see POST /api/settings/hide-model). Non-admin callers never
    # receive a hidden entry at all — this field only ever comes back
    # `true` for an admin, so their own catalog view can show which
    # models they've hidden and offer an "Unhide" action.
    hidden: bool = False
    # Whether "Remove from list" should be offered (admin-only, see DELETE /api/settings/model-catalog/extended)
    # — true only for a not-yet-installed entry from the admin-managed extended catalog (see
    # app.services.extended_model_catalog_service.ExtendedModelCatalog.build); always False for a hand-curated
    # app/model_catalog.py default (there's nothing stored to delete) and for an installed model (that gets
    # "Uninstall" instead — removing it from a *list* while it's still actually pulled into the ML engine would just be
    # confusing, and re-adding it later would need a redundant Hugging Face re-verification for no reason).
    removable: bool = False
    # Whether this model's architecture+quantization is actually implemented by the Matricxon engine (see
    # ../matricxon/app/architectures/registry.py and .../gguf/dequant/registry.py) — independent of Ollama, which
    # can run every entry this catalog offers. For a hand-curated app/model_catalog.py entry, hand-verified True
    # or False against those two registries (see that entry's own comment for which and why). Defaults True here
    # only as the field's bare fallback; an admin-added extended-catalog entry (arbitrary Hugging Face repo, no
    # verified architecture) is explicitly constructed with this False instead — see
    # app.services.extended_model_catalog_service.ExtendedModelCatalog.build's own comment — since Matricxon fails
    # closed on an unrecognized architecture and a false "supported" badge would be actively misleading.
    # A vision-projector (mmproj) sidecar, pulled alongside its own paired text model rather than run as a
    # standalone chat model — see
    # app.services.huggingface_client.HuggingFaceCatalogSearch.is_projector_file's own docstring. Lets the
    # frontend skip the matricxon_supported "Not supported" badge for one of these specifically: that badge
    # means "Matricxon can't run this as a chat model," which is a true but actively misleading thing to say
    # about a file that was never meant to be run as one on its own — confirmed live, it read as "Matricxon
    # can't handle this," when the paired text model this projector exists to be paired with may well be
    # supported on its own (see MatricxonSupportChecker.verdict_for's own docstring on is_projector).
    is_projector: bool = False
    # True for an installed model that isn't in app/model_catalog.py's hand-curated list at all — surfaced
    # through ChatModelCatalogBuilder._auto_discovered_entry, InstalledProjectorCatalog.entries, or
    # ExtendedModelCatalog.build (the latter always paired with installed=False, so it never actually changes
    # what the frontend shows there). Lets the frontend label a stray/manually-added install ("Not in catalog")
    # instead of it looking like a hand-curated entry with broken/missing data — confirmed live (2026-09-21)
    # that a duplicate model pulled under an unfamiliar repo name showed up as a bare "Phi2 · unknown" row with
    # nothing explaining why it looked so different from every other entry.
    is_auto_discovered: bool = False
    # A diffusion model installed into the image engine's own store (see app.services.image_model_service), not
    # an Ollama/Matricxon chat model: the frontend offers Pull/Uninstall (via /api/settings/image-models) but no
    # Select/Disable, and no chat-engine support badge.
    is_image: bool = False

    # --- Backward-compatible views onto engine_support above -----------------------------------------------
    # Computed, not settable at construction time — every producer builds engine_support directly instead (see
    # app.services.engine_support_checker.EngineSupportSet.support_for). Kept only so old callers/tests reading
    # entry.matricxon_supported (etc.) keep working unchanged; new code should read engine_support directly.
    @computed_field  # type: ignore[prop-decorator]
    @property
    def min_ram_gb_ollama(self) -> float | None:
        return self.engine_support.get("ollama", EngineModelSupport()).min_ram_gb

    @computed_field  # type: ignore[prop-decorator]
    @property
    def min_ram_gb_matricxon(self) -> float | None:
        return self.engine_support.get("matricxon", EngineModelSupport()).min_ram_gb

    @computed_field  # type: ignore[prop-decorator]
    @property
    def matricxon_supported(self) -> bool:
        return self.engine_support.get("matricxon", EngineModelSupport()).supported

    @computed_field  # type: ignore[prop-decorator]
    @property
    def matricxon_unsupported_reason(self) -> str | None:
        return self.engine_support.get("matricxon", EngineModelSupport()).reason


class ModelCatalogResponse(BaseModel):
    entries: list[CatalogEntry]
    hardware: HardwareSummary
    # The active engine's own EngineCapabilities.model_management (see app.services.engines.base) - False for a
    # connector like RunPod, which manages its own model(s) remotely and has nothing here to Pull/Uninstall/Hide,
    # nor a Hugging Face catalog to browse. The Model tab greys out those actions instead of rendering ones that
    # would just fail or do nothing.
    model_management: bool = True


class EmbeddingCatalogEntry(BaseModel):
    """One embedding model from app/model_catalog.py's EMBEDDING_CATALOG — the embedding-model analogue of
    CatalogEntry above, kept as its own type rather than reusing that one since several of its fields (hidden,
    removable, text_capable, vision) are meaningless for an embedding-only model."""

    family: str
    vendor: str
    tag: str
    parameter_size: str
    context_length: int | None = None
    # Output vector size (768 for nomic-embed-text, 384 for MiniLM) — not meaningful for a chat model, so kept
    # here rather than added to CatalogEntry.
    embedding_dim: int
    download_gb: float | None = None
    min_ram_gb: float
    # See CatalogEntry.engine_support's own docstring — same meaning here.
    engine_support: dict[EngineName, EngineModelSupport] = Field(default_factory=dict)
    installed: bool
    hardware_ok: bool
    unavailable_reason: str | None = None

    # See CatalogEntry's identical computed properties — same "old field name, new dict underneath" reasoning.
    @computed_field  # type: ignore[prop-decorator]
    @property
    def min_ram_gb_ollama(self) -> float | None:
        return self.engine_support.get("ollama", EngineModelSupport()).min_ram_gb

    @computed_field  # type: ignore[prop-decorator]
    @property
    def min_ram_gb_matricxon(self) -> float | None:
        return self.engine_support.get("matricxon", EngineModelSupport()).min_ram_gb

    @computed_field  # type: ignore[prop-decorator]
    @property
    def matricxon_supported(self) -> bool:
        return self.engine_support.get("matricxon", EngineModelSupport()).supported

    @computed_field  # type: ignore[prop-decorator]
    @property
    def matricxon_unsupported_reason(self) -> str | None:
        return self.engine_support.get("matricxon", EngineModelSupport()).reason


class EmbeddingModelCatalogResponse(BaseModel):
    entries: list[EmbeddingCatalogEntry]
    # Whichever tag app.config.EMBEDDING_MODEL currently resolves to, so the UI can label which entry is
    # actually in use without every entry needing its own is_default flag.
    default_tag: str
    hardware: HardwareSummary


class InstalledModelsResponse(BaseModel):
    """GET /api/settings/installed-models — the chat page's model-switch
    picker on an existing conversation, which only needs plain tags, not
    get_model_catalog's hardware/download info."""

    models: list[str]
