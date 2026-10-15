"""Shapes for Settings > Model: the default catalog (app/model_catalog.py), the admin-managed extended
catalog (extended_model_catalog_service.py) and Hugging Face search (huggingface_client.py).
Split from schemas/settings.py for the file-size cap.
"""

from pydantic import BaseModel, Field, computed_field

from app.schemas.common import EngineName
from app.schemas.settings import HardwareSummary


class EngineModelSupport(BaseModel):
    """One engine's verdict on one model (engine_support_checker.SupportVerdict / EngineSupportChecker.estimated_ram_gb,
    via EngineSupportSet.support_for). One dict keyed by EngineName replaces the per-engine fields (kept as
    computed properties), so a new engine needs no new field.
    """

    supported: bool = True
    reason: str | None = None
    # This engine's own per-model RAM estimate, or None if unknown or not applicable (never extrapolated from another
    # engine).
    min_ram_gb: float | None = None


class CompanionFile(BaseModel):
    """An extra file a split image model needs (text encoder, VAE); offered, never downloaded automatically."""

    flag: str
    label: str
    size_gb: float
    installed: bool


class CatalogEntry(BaseModel):
    """One model Settings can offer: already pulled, pullable and compatible, or hardware-blocked
    (source: app/model_catalog.py; `installed`/`hardware_ok`: settings_service.py).
    """

    family: str
    vendor: str
    tag: str | None
    parameter_size: str
    context_length: int | None = None
    download_gb: float | None = None
    # Hardware-gating threshold for a not-yet-installed entry (ChatModelCatalogBuilder hardware_ok, routers/settings.py
    # pull check): app/model_catalog.py's hand-typed figure. For display prefer min_ram_gb_ollama/min_ram_gb_matricxon,
    # which are computed per engine (Matricxon dequantizes to bf16, so a Ministral-3B Q4_K_M needs ~7.5GB, not ~3GB,
    # ).
    min_ram_gb: float
    # Every engine's verdict on this model (EngineModelSupport), built once per catalog build by
    # EngineSupportSet.support_for. The min_ram_gb_*/matricxon_supported/... properties read it for old callers;
    # new code should read the dict.
    engine_support: dict[EngineName, EngineModelSupport] = Field(default_factory=dict)
    locally_runnable: bool
    installed: bool
    hardware_ok: bool
    unavailable_reason: str | None = None
    # Whether the model can see an image attachment (chat_attachment_service, "Default vision model" picker).
    # For a not-yet-installed curated entry it is a hand-annotated estimate; once installed it is read live
    # from the active engine's "vision" tag (the curated flag mirrors Ollama's projector auto-pairing and can
    # disagree under Matricxon).
    vision: bool = False
    # System-tab roles this model fills (DefaultModelSettings.chat_roles); non-empty shows "In use".
    in_use_as: list[str] = Field(default_factory=list)
    # Whether the model can also hold a text conversation: true for every entry (they pass ChatModelCatalogBuilder's
    # "completion" filter), so only `vision and text_capable` ("+Vision") occurs; `vision and not text_capable` is
    # modeled but unreachable.
    text_capable: bool = True
    # Whether the active engine has no confirmed chat format for this model (Matricxon's "chat_format_unverified"
    # tag,../matricxon/app/models/capabilities.py: a "completion" model with no chat_template and no mistral3
    # tokenizer falls back to a guess). Read live post-install only (nothing pre-install reveals it). Flags models that
    # may not behave like chat models (a Hebrew Mistral fine-tune answered incoherently).
    chat_format_unverified: bool = False
    # Whether an admin hid this model from regular users (POST /api/settings/hide-model). Hidden entries are never
    # sent to non-admins; for admins it is true so they can offer "Unhide".
    hidden: bool = False
    # Whether to offer "Remove from list" (admin-only, DELETE /api/settings/model-catalog/extended): true only for
    # a not-yet-installed extended-catalog entry (ExtendedModelCatalog.build). Never for curated defaults (nothing
    # stored) or installed models (they get "Uninstall").
    removable: bool = False
    # Whether Matricxon implements this architecture+quantization (../matricxon/app/architectures/registry.py,
    # .../gguf/dequant/registry.py); Ollama runs every entry. Curated entries are hand-verified. Defaults True as a
    # bare fallback; admin-added extended entries (arbitrary repos) are built False (ExtendedModelCatalog.build),
    # since Matricxon fails closed on unknown architectures and a false "supported" badge would mislead.
    # A vision-projector (mmproj) sidecar, pulled with its paired text model, not run as a chat model
    # (HuggingFaceCatalogSearch.is_projector_file). Lets the frontend skip the "Not supported" badge for it, which
    # would mislead: the paired model may be supported (MatricxonSupportChecker.verdict_for, is_projector).
    is_projector: bool = False
    # True for an installed model absent from the curated list (ChatModelCatalogBuilder._auto_discovered_entry,
    # InstalledProjectorCatalog.entries, ExtendedModelCatalog.build, the last always with installed=False). Labels it
    # "Not in catalog" instead of looking like a curated entry with broken data (a bare "Phi2 · unknown" row,
    # ).
    is_auto_discovered: bool = False
    # A diffusion model in the image engine's store (image_model_service), not a chat model: Pull/Uninstall via
    # /api/settings/image-models, no Select/Disable, no engine support badge.
    is_image: bool = False
    # Image models only: extra files needed beside the weights, and whether each is on disk.
    companions: list[CompanionFile] = Field(default_factory=list)

    # Backward-compatible views onto engine_support
    # Computed, not settable: producers build engine_support (EngineSupportSet.support_for). Kept for old
    # callers/tests; new code should read engine_support.
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
    # The active engine's EngineCapabilities.model_management (engines.base): False for connectors like RunPod,
    # which manage models remotely, so the Model tab greys out Pull/Uninstall/Hide and the Hugging Face browser.
    model_management: bool = True


class EmbeddingCatalogEntry(BaseModel):
    """One embedding model from EMBEDDING_CATALOG; separate from CatalogEntry since hidden, removable,
    text_capable and vision mean nothing for embeddings.
    """

    family: str
    vendor: str
    tag: str
    parameter_size: str
    context_length: int | None = None
    # Output vector size (768 nomic-embed-text, 384 MiniLM).
    embedding_dim: int
    download_gb: float | None = None
    min_ram_gb: float
    # See CatalogEntry.engine_support.
    engine_support: dict[EngineName, EngineModelSupport] = Field(default_factory=dict)
    installed: bool
    hardware_ok: bool
    unavailable_reason: str | None = None
    # Chosen under Admin settings > System > Default embedding model.
    in_use: bool = False

    # See CatalogEntry's computed properties.
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
    # The tag app.config.EMBEDDING_MODEL resolves to, so the UI marks the one in use.
    default_tag: str
    hardware: HardwareSummary


class InstalledModelsResponse(BaseModel):
    """GET /api/settings/installed-models: plain tags for the chat page's model switcher."""

    models: list[str]
