"""The one place every "which model is the default" read/write goes through, replacing three scattered pairs
in model_catalog_service.py: default for new accounts, default vision model (image-attachment replies) and
default embedding model (RAG). Isolated from ChatModelCatalogBuilder/EmbeddingModelCatalogService (which
depend on it, not the reverse), so a change to reading, validation or fallback is made once.

Not settings_service.get_default_model/set_default_model: that is a per-user preference for the user's own
new conversations, separate from these three system-wide defaults.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import SYSTEM_OWNER_ID, AppSetting
from app.services.inference_client import list_models

_VISION_KEY = "default_vision_model"
_EMBEDDING_KEY = "default_embedding_model"


class DefaultModelSettings:
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
        """The admin-configured model for new accounts (user_service.create_user); falls back to the first installed
        model if unset or since uninstalled. `installed` is passed in since callers already have it.
        """
        configured = await self._get(self.FOR_NEW_USERS_KEY)
        if configured in installed:
            return configured
        return installed[0] if installed else None

    async def set_for_new_users(self, model: str) -> None:
        await self._set(self.FOR_NEW_USERS_KEY, model)

    async def vision(self) -> str | None:
        """The admin-configured model that answers any message with an image attachment, for that reply only
        (chat_service.build_reply_stream never changes the conversation's `model`, so the next message reverts).
        Same installed/fallback contract as for_new_users; None if no vision model is installed.
        """
        installed = await self._installed_with_capability("vision")
        configured = await self._get(_VISION_KEY)
        if configured in installed:
            return configured
        return installed[0] if installed else None

    async def set_vision(self, model: str) -> None:
        await self._set(_VISION_KEY, model)

    async def embedding(self) -> str | None:
        """The model inference_client.embed uses for the Knowledge base
        (document_ingest/document_retrieval/document_sync),
        replacing the fixed app.config.EMBEDDING_MODEL. Same fallback as vision. None if no embedding model is
        installed; callers degrade as before (RAG silently unavailable, no hard error).
        """
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
