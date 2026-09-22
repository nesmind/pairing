"""
The one place every "which model is the default" read/write goes through — consolidates what used to be three
separate, independently-maintained function pairs scattered across app/services/model_catalog_service.py
(default model for brand-new user accounts, default vision model for an image-attachment reply, default
embedding model for the RAG knowledge base). Deliberately its own isolated module, not folded into
ChatModelCatalogBuilder/EmbeddingModelCatalogService (which each depend on this, never the other way around):
a future change to how a default is read/validated/falls back can now only be made in one place, not silently
missed in one of three near-identical copies.

Not the same setting as app.services.settings_service.get_default_model/set_default_model — that one is a
*per-user* "which model my own new conversations start with" preference, unrelated to any of the three
system-wide defaults here, and stays exactly where and what it is today.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import SYSTEM_OWNER_ID, AppSetting
from app.services.inference_client import list_models

_VISION_KEY = "default_vision_model"
_EMBEDDING_KEY = "default_embedding_model"


class DefaultModelSettings:
    # Public (unlike the other two keys above) since app.services.engine_switch_service.clear_stale_default_models
    # needs to identify this exact AppSetting row alongside settings_service.DEFAULT_MODEL_KEY, to clear it out
    # when its configured tag stops being installed after an engine switch.
    FOR_NEW_USERS_KEY = "default_model_for_new_users"

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def _get(self, key: str) -> str | None:
        row = await self._db.get(AppSetting, (SYSTEM_OWNER_ID, key))
        return row.value["model"] if row else None

    async def _set(self, key: str, model: str) -> None:
        row = await self._db.get(AppSetting, (SYSTEM_OWNER_ID, key))
        value = {"model": model}
        if row is None:
            self._db.add(AppSetting(owner_id=SYSTEM_OWNER_ID, key=key, value=value))
        else:
            row.value = value
        await self._db.commit()

    async def for_new_users(self, installed: list[str]) -> str | None:
        """The admin-configured model new accounts are created with (see app.services.user_service.create_user)
        — falls back to whichever installed model comes first if never configured, or if the configured one
        was since uninstalled. `installed` is passed in rather than fetched here since every caller already has
        it (avoids a redundant engine round trip)."""
        configured = await self._get(self.FOR_NEW_USERS_KEY)
        if configured in installed:
            return configured
        return installed[0] if installed else None

    async def set_for_new_users(self, model: str) -> None:
        await self._set(self.FOR_NEW_USERS_KEY, model)

    async def vision(self) -> str | None:
        """The admin-configured model any message with an image attachment is answered by, for that one reply
        only — see app.services.chat_service.build_reply_stream, which never changes the conversation's own
        `model` column for this, so the very next message (with no image) reverts automatically. Same
        installed/fallback contract as for_new_users above; None if no vision-capable model is installed at
        all."""
        installed = await self._installed_with_capability("vision")
        configured = await self._get(_VISION_KEY)
        if configured in installed:
            return configured
        return installed[0] if installed else None

    async def set_vision(self, model: str) -> None:
        await self._set(_VISION_KEY, model)

    async def embedding(self) -> str | None:
        """The admin-configured model app.services.inference_client.embed calls use for the Knowledge base
        (RAG) feature — see app.services.document_ingest/document_retrieval/document_sync, every one of which
        reads this instead of the old fixed app.config.EMBEDDING_MODEL constant. Same fallback pattern as
        vision() above. None if no embedding model is installed at all — every caller treats that the same way
        "no embedding model" already degraded before this was configurable (RAG silently unavailable, not a
        hard error)."""
        installed = await self._installed_with_capability("embedding")
        configured = await self._get(_EMBEDDING_KEY)
        if configured in installed:
            return configured
        return installed[0] if installed else None

    async def set_embedding(self, model: str) -> None:
        await self._set(_EMBEDDING_KEY, model)

    @staticmethod
    async def _installed_with_capability(capability: str) -> list[str]:
        installed = await list_models()
        return [m["name"] for m in installed if capability in m.get("capabilities", [])]
