"""
Multi-instance-safe in-process cache of every Connector's admin-configured state
(see app.services.connector_config_service for the DB-backed truth) — the same
write-through pattern app.services.engine_service already uses for the active
engine: load_cache_from_db() once at startup (see app.services.startup_service) and
again from every other local instance's own internal-refresh handler (see
app/routers/connectors.py) after an admin saves config or flips enabled, then plain
synchronous reads for the actual hot-path calls (see app.services.runpod_client,
which reads a connector's config on every request — no db session is ever threaded
through an InferenceEngine method).
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.services import connector_config_service
from app.services.connectors import CONNECTORS

_cache: dict[str, dict] = {}


async def load_cache_from_db(db: AsyncSession) -> None:
    """Populates the in-process cache from the DB for every registered connector —
    call once at startup (app.services.startup_service.run_startup_tasks, on every
    instance) and from every connector's own internal-refresh handler."""
    global _cache
    fresh: dict[str, dict] = {}
    for connector in CONNECTORS:
        config = await connector_config_service.get_config(db, connector.id)
        enabled = await connector_config_service.get_enabled(db, connector.id)
        configured = all(config.get(f.name) for f in connector.config_fields if f.required)
        last_test_passed, _ = await connector_config_service.get_test_result(db, connector.id)
        fresh[connector.id] = {
            "config": config,
            "enabled": enabled,
            "configured": configured,
            "last_test_passed": last_test_passed,
        }
    _cache = fresh


def get_config(connector_id: str) -> dict[str, str]:
    """This connector's live, decrypted config fields — empty dict if never saved or
    never loaded (e.g. an unknown id)."""
    return _cache.get(connector_id, {}).get("config", {})


def is_enabled(connector_id: str) -> bool:
    return bool(_cache.get(connector_id, {}).get("enabled"))


def is_ready(connector_id: str) -> bool:
    """Enabled, every required field has a saved value, AND the last "Test connection" action against that
    exact saved config actually succeeded (see connector_config_service.set_config's own docstring on why any
    config edit resets this) — what app.services.engines.runpod_engine.RunPodEngine.is_ready() actually checks."""
    entry = _cache.get(connector_id, {})
    return bool(entry.get("enabled")) and bool(entry.get("configured")) and bool(entry.get("last_test_passed"))
