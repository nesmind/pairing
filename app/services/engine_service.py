"""
Which inference engine — Ollama or Matricxon — app.services.inference_client dispatches chat/embedding/model-
management calls to right now (see Settings > External servers' "Active engine" picker, app/routers/engine_admin.py).

Both engines are configured, started, and stopped fully independently (see app.services.ollama_admin/
matricxon_admin routers and the ollama_pool/matricxon_pool each one's own config feeds) — this module owns only
the one extra bit of state that decides which of the two actually serves live traffic. Cached in-process (not
re-read from the DB on every call — this sits on the hot chat-generation path, same reasoning as
app.services.ollama_pool's own cached host list) via a write-through pattern: load_cache_from_db() once at
startup (see app.services.startup_service), then current_engine() for every subsequent hot-path read; set_active_engine
keeps the cache and the DB in lockstep on every admin save.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import SYSTEM_OWNER_ID, AppSetting
from app.schemas import EngineName

ACTIVE_ENGINE_KEY = "active_llm_engine"
# Matricxon is the default for now (see Settings > External servers' "Active engine" picker) while it's still
# the engine under active development/testing — revisit once Ollama should go back to being the safer default.
DEFAULT_ENGINE: EngineName = "matricxon"

_cached_engine: EngineName = DEFAULT_ENGINE


async def get_active_engine(db: AsyncSession) -> EngineName:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, ACTIVE_ENGINE_KEY))
    return row.value["engine"] if row else DEFAULT_ENGINE


async def set_active_engine(db: AsyncSession, engine: EngineName) -> None:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, ACTIVE_ENGINE_KEY))
    value = {"engine": engine}
    if row is None:
        db.add(AppSetting(owner_id=SYSTEM_OWNER_ID, key=ACTIVE_ENGINE_KEY, value=value))
    else:
        row.value = value
    await db.commit()
    global _cached_engine
    _cached_engine = engine


async def load_cache_from_db(db: AsyncSession) -> None:
    """Populates the in-process cache from the DB — call once at startup (see
    app.services.startup_service.run_startup_tasks) and from every other local instance's own
    internal-refresh handler (see app/routers/engine_admin.py), the same "each process re-reads the shared DB
    for itself" pattern app.services.ollama_pool's own broadcast refresh already uses."""
    global _cached_engine
    _cached_engine = await get_active_engine(db)


def current_engine() -> EngineName:
    """Synchronous, hot-path read of the cached active engine — see app.services.inference_client, the only
    caller that needs this on every chat/embedding/model-management call."""
    return _cached_engine
