"""Which engine, Ollama or Matricxon, inference_client dispatches chat/embedding/model-management calls to
(Settings > External servers' "Active engine" picker, routers/engine_admin.py).

Both are configured, started and stopped independently (ollama_admin/matricxon_admin, their pools); this
module owns only the state choosing which serves traffic. Cached in-process (hot chat path, like
ollama_pool's host list) write-through: load_cache_from_db at startup (startup_service), then
current_engine; set_active_engine keeps cache and DB in step.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import SYSTEM_OWNER_ID, AppSetting

# DEFAULT_ENGINE is re-exported (not redefined) so existing `engine_service.DEFAULT_ENGINE` uses and tests
# that reset _cached_engine keep working (see schemas.common.DEFAULT_ENGINE for why).
from app.schemas import DEFAULT_ENGINE, EngineName

ACTIVE_ENGINE_KEY = "active_llm_engine"

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
    """Loads the cache from the DB; call at startup (startup_service.run_startup_tasks) and from each instance's
    internal-refresh handler (routers/engine_admin.py), as in ollama_pool's broadcast refresh.
    """
    global _cached_engine
    _cached_engine = await get_active_engine(db)


def current_engine() -> EngineName:
    """Synchronous hot-path read of the cached active engine (inference_client calls it on every
    chat/embedding/model-management call)."""
    return _cached_engine
