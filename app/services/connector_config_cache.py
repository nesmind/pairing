"""Multi-instance-safe in-process cache of each Connector's admin state (DB truth: connector_config_service).
Write-through like engine_service's active engine: load_cache_from_db at startup (startup_service) and
from each instance's internal-refresh handler (routers/connectors.py) after a save or enable change; hot
paths (runpod_client reads config per request, with no db session threaded through InferenceEngine)
then read synchronously.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.services import connector_config_service
from app.services.connectors import CONNECTORS

_cache: dict[str, dict] = {}


async def load_cache_from_db(db: AsyncSession) -> None:
    """Loads every registered connector's state from the DB; call at startup (run_startup_tasks, every
    instance) and from each connector's internal-refresh handler.
    """
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
    """The connector's live decrypted config; empty if never saved or loaded (e.g. unknown id)."""
    return _cache.get(connector_id, {}).get("config", {})


def is_enabled(connector_id: str) -> bool:
    return bool(_cache.get(connector_id, {}).get("enabled"))


def is_ready(connector_id: str) -> bool:
    """Enabled, all required fields saved, AND the last "Test connection" on that saved config succeeded
    (any edit resets it, see connector_config_service.set_config). What RunPodEngine.is_ready checks.
    """
    entry = _cache.get(connector_id, {})
    return bool(entry.get("enabled")) and bool(entry.get("configured")) and bool(entry.get("last_test_passed"))
