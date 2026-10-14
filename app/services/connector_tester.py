"""Test connection for a connector's form values, before or after they are saved."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas import ConnectorTestResult
from app.services import connector_config_cache, connector_config_service
from app.services.connectors import get_connector


class ConnectorTester:
    """Tests the form values merged over the saved config. Only a test of the saved config is recorded (and can make
    the connector ready); a test of unsaved values just reports its result."""

    def __init__(self, db: AsyncSession, connector_id: str) -> None:
        self._db = db
        self._connector = get_connector(connector_id)  # KeyError for an unknown id

    async def run(self, values: dict[str, str]) -> ConnectorTestResult:
        # Imported here: engines.registry sits above connectors (see routers/connectors._to_out).
        from app.services.engines.registry import registry

        connector_id = self._connector.id
        saved = await connector_config_service.get_config(self._db, connector_id)
        candidate = connector_config_service.merge_values(connector_id, saved, values)
        passed, message = await registry.get(self._connector.engine_name).test_connection(candidate)
        is_saved = candidate == saved
        if is_saved:
            await connector_config_service.set_test_result(self._db, connector_id, passed, message)
            await connector_config_cache.load_cache_from_db(self._db)
        return ConnectorTestResult(passed=passed, message=message, saved=is_saved)
