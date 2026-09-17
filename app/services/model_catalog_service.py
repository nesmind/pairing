"""
Everything about which Ollama models exist and which ones a given user
is allowed to see/pick: the curated catalog (app/model_catalog.py) cross-
referenced with what's actually installed, the admin-configurable
hidden-tags list, and the admin-configured default for brand-new
accounts. See app/services/settings_service.py for simple per-user
key/value settings instead.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app import hardware, model_catalog
from app.config import EMBEDDING_MODEL
from app.models import SYSTEM_OWNER_ID, AppSetting, User
from app.schemas import CatalogEntry, EmbeddingCatalogEntry, EmbeddingModelCatalogResponse, ModelCatalogResponse
from app.services.ollama_client import list_models

HIDDEN_MODELS_KEY = "hidden_model_tags"
DEFAULT_MODEL_FOR_NEW_USERS_KEY = "default_model_for_new_users"
DEFAULT_VISION_MODEL_KEY = "default_vision_model"


async def installed_chat_models() -> list[str]:
    """Tags of every model actually pulled into Ollama that can hold a
    conversation — same "completion" capability filter build_model_catalog
    uses to decide what belongs in the chat model picker at all, reused
    here so "is there anything to assign a new user" means the same
    thing in both places."""
    installed = await list_models()
    return [m["name"] for m in installed if "completion" in m.get("capabilities", [])]


async def installed_vision_models() -> list[str]:
    """Tags of every installed model that can accept an image as part of
    a chat request (Ollama's own "vision" capability flag on /api/tags) —
    used for the Settings > System "default vision model" picker (see
    get_default_vision_model below), never for the ordinary chat model
    picker, which only ever offers "completion" models."""
    installed = await list_models()
    return [m["name"] for m in installed if "vision" in m.get("capabilities", [])]


async def get_hidden_tags(db: AsyncSession) -> set[str]:
    """Tags an admin has hidden from regular users' model picker (see
    set_hidden_tags below). System-wide, not per-user — hiding a model
    is a catalog-curation decision, not a personal preference."""
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, HIDDEN_MODELS_KEY))
    return set(row.value["tags"]) if row else set()


async def set_hidden_tags(db: AsyncSession, tags: set[str]) -> None:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, HIDDEN_MODELS_KEY))
    value = {"tags": sorted(tags)}
    if row is None:
        db.add(AppSetting(owner_id=SYSTEM_OWNER_ID, key=HIDDEN_MODELS_KEY, value=value))
    else:
        row.value = value
    await db.commit()


async def get_default_model_for_new_users(db: AsyncSession, installed: list[str]) -> str | None:
    """The admin-configured model new accounts are created with (see
    app.services.user_service.create_user) — falls back to whichever
    installed model comes first if never configured, or if the
    configured one was since uninstalled. `installed` is passed in
    rather than fetched here since every caller already has it (avoids a
    redundant Ollama round trip)."""
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, DEFAULT_MODEL_FOR_NEW_USERS_KEY))
    configured = row.value["model"] if row else None
    if configured in installed:
        return configured
    return installed[0] if installed else None


async def set_default_model_for_new_users(db: AsyncSession, model: str) -> None:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, DEFAULT_MODEL_FOR_NEW_USERS_KEY))
    value = {"model": model}
    if row is None:
        db.add(AppSetting(owner_id=SYSTEM_OWNER_ID, key=DEFAULT_MODEL_FOR_NEW_USERS_KEY, value=value))
    else:
        row.value = value
    await db.commit()


async def get_default_vision_model(db: AsyncSession) -> str | None:
    """The admin-configured model any message with an image attachment
    is answered by, for that one reply only — see
    app.services.chat_service.build_reply_stream, which never changes the
    conversation's own `model` column for this, so the very next message
    (with no image) reverts automatically. Falls back to whichever
    installed vision model comes first if never configured, or if the
    configured one was since uninstalled — same pattern as
    get_default_model_for_new_users above. None if no vision model is
    installed at all."""
    installed = await installed_vision_models()
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, DEFAULT_VISION_MODEL_KEY))
    configured = row.value["model"] if row else None
    if configured in installed:
        return configured
    return installed[0] if installed else None


async def set_default_vision_model(db: AsyncSession, model: str) -> None:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, DEFAULT_VISION_MODEL_KEY))
    value = {"model": model}
    if row is None:
        db.add(AppSetting(owner_id=SYSTEM_OWNER_ID, key=DEFAULT_VISION_MODEL_KEY, value=value))
    else:
        row.value = value
    await db.commit()


async def build_model_catalog(db: AsyncSession, user: User) -> ModelCatalogResponse:
    """Every model Settings can offer: the curated list in
    app/model_catalog.py, plus anything chat-capable already pulled into
    Ollama that isn't in that list (e.g. pulled manually on the
    terminal) so the picker never hides something the user already has.
    Each entry says whether it's installed and whether this machine's
    hardware (app/hardware.py) can actually run it — embedding-only
    models like app.config.EMBEDDING_MODEL are excluded, same as
    before, since they can't hold a conversation.

    A model an admin has hidden (see set_hidden_tags above) is left out
    of the response entirely for a regular user — not just marked hidden
    client-side — so there's no way to discover a hidden tag by
    inspecting the API response. An admin gets every entry, including
    hidden ones (flagged via `hidden`), so they can unhide."""
    installed_models = await list_models()

    is_admin = user.role == "admin"
    hidden_tags = await get_hidden_tags(db)

    installed_by_tag = {m["name"]: m for m in installed_models if "completion" in m.get("capabilities", [])}
    capacity_gb = hardware.available_capacity_gb()

    entries = []
    for entry in model_catalog.CATALOG:
        tag = entry["tag"]
        if tag in hidden_tags and not is_admin:
            continue
        installed = tag is not None and tag in installed_by_tag
        entries.append(
            CatalogEntry(
                family=entry["family"],
                vendor=entry.get("vendor", "Other"),
                tag=tag,
                parameter_size=entry["parameter_size"],
                context_length=entry.get("context_length"),
                download_gb=entry.get("download_gb"),
                min_ram_gb=entry["min_ram_gb"],
                locally_runnable=entry["locally_runnable"],
                installed=installed,
                # Already-installed models get a pass on the live check —
                # they clearly ran well enough to be downloaded before, and
                # re-blocking something the user already has would be a
                # worse experience than trusting past success.
                hardware_ok=installed or (entry["locally_runnable"] and capacity_gb >= entry["min_ram_gb"]),
                unavailable_reason=entry.get("unavailable_reason"),
                hidden=tag in hidden_tags,
                vision=entry.get("vision", False),
                # Every CATALOG entry is a normal chat model by
                # definition (this whole list is curated as one) — see
                # CatalogEntry.text_capable's own docstring for why this
                # is never actually False here today.
                text_capable=True,
            )
        )

    catalog_tags = {entry["tag"] for entry in model_catalog.CATALOG if entry["tag"]}
    for name, installed_model in installed_by_tag.items():
        if name in catalog_tags:
            continue
        if name in hidden_tags and not is_admin:
            continue
        details = installed_model.get("details", {})
        size_gb = (installed_model.get("size") or 0) / 1_000_000_000
        entries.append(
            CatalogEntry(
                family=(details.get("family") or "Other").title(),
                vendor="Other",
                tag=name,
                parameter_size=details.get("parameter_size") or "?",
                context_length=details.get("context_length"),
                download_gb=round(size_gb, 1) if size_gb else None,
                min_ram_gb=round(size_gb * 1.25, 1) if size_gb else 0,
                locally_runnable=True,
                installed=True,
                hardware_ok=True,
                unavailable_reason=None,
                hidden=name in hidden_tags,
                # Already installed, so both of these read Ollama's own
                # real capability tags for it, not a hand-curated guess
                # like the CATALOG branch above — though text_capable is
                # always True in practice here too, since installed_by_tag
                # (this loop's own source) already filtered to
                # "completion"-capable models only.
                vision="vision" in installed_model.get("capabilities", []),
                text_capable="completion" in installed_model.get("capabilities", []),
            )
        )

    return ModelCatalogResponse(entries=entries, hardware=hardware.hardware_summary())


async def get_installed_models_for_user(db: AsyncSession, user: User) -> list[str]:
    """Installed chat-capable model tags a conversation can be switched
    to — used by the chat page's model-switch picker. Admin-hidden tags
    are left out for a regular user, same visibility rule as
    build_model_catalog; an already-hidden model an existing
    conversation is using isn't affected by that (see set_hidden_tags's
    docstring in app/routers/settings.py's hide-model endpoint)."""
    installed = await installed_chat_models()
    if user.role == "admin":
        return installed
    hidden_tags = await get_hidden_tags(db)
    return [m for m in installed if m not in hidden_tags]


async def build_embedding_model_catalog() -> EmbeddingModelCatalogResponse:
    """Every embedding model in app/model_catalog.py's EMBEDDING_CATALOG, cross-referenced with what's
    actually pulled into Ollama — the embedding-model analogue of build_model_catalog above, kept separate
    since EMBEDDING_CATALOG has no hidden-tags/per-user logic worth sharing. Open to any logged-in user, same
    visibility as GET /api/settings/rag-availability — installed status isn't admin-secret, only pulling is."""
    installed_models = await list_models()
    installed_by_tag = {m["name"]: m for m in installed_models if "embedding" in m.get("capabilities", [])}
    capacity_gb = hardware.available_capacity_gb()

    entries = []
    for entry in model_catalog.EMBEDDING_CATALOG:
        tag = entry["tag"]
        installed = tag in installed_by_tag
        entries.append(
            EmbeddingCatalogEntry(
                family=entry["family"],
                vendor=entry["vendor"],
                tag=tag,
                parameter_size=entry["parameter_size"],
                context_length=entry.get("context_length"),
                embedding_dim=entry["embedding_dim"],
                download_gb=entry.get("download_gb"),
                min_ram_gb=entry["min_ram_gb"],
                installed=installed,
                hardware_ok=installed or capacity_gb >= entry["min_ram_gb"],  # same exemption as above
                unavailable_reason=entry.get("unavailable_reason"),
            )
        )

    return EmbeddingModelCatalogResponse(
        entries=entries, default_tag=EMBEDDING_MODEL, hardware=hardware.hardware_summary()
    )
