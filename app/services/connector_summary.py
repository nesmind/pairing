"""The summary the Inference engines tab shows when a connector engine is picked (GET /api/connectors/{id}/summary):
its state, non-secret settings, model counts and, where the provider reports it, usage."""

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas import ConnectorSummary, ConnectorUsage, SummaryRow
from app.services import connector_config_cache, connector_config_service
from app.services.connectors import get_connector
from app.services.engines.registry import registry

logger = logging.getLogger("llama_chat")


class ConnectorSummaryBuilder:
    def __init__(self, db: AsyncSession, connector_id: str) -> None:
        self._db = db
        self._connector = get_connector(connector_id)  # KeyError for an unknown id
        self._engine = registry.get(self._connector.engine_name)

    async def _settings_rows(self) -> list[SummaryRow]:
        """Non-secret fields only; a secret shows as saved or not."""
        values = await connector_config_service.get_config_for_display(self._db, self._connector.id)
        rows = []
        for field in self._connector.config_fields:
            if field.secret:
                rows.append(SummaryRow(label=field.label, value="Saved" if values.get(f"has_{field.name}") else "—"))
            elif values.get(field.name):
                rows.append(SummaryRow(label=field.label, value=str(values[field.name])))
        return rows

    async def _model_counts(self, summary: ConnectorSummary) -> None:
        try:
            models = await self._engine.list_models()
        except registry.all_error_types() as exc:
            summary.models_error = str(exc)
            return
        summary.chat_models = sum("completion" in m.get("capabilities", []) for m in models)
        summary.embedding_models = sum("embedding" in m.get("capabilities", []) for m in models)

    async def build(self) -> ConnectorSummary:
        connector_id = self._connector.id
        passed, message = await connector_config_service.get_test_result(self._db, connector_id)
        summary = ConnectorSummary(
            id=connector_id,
            display_name=self._connector.display_name,
            enabled=await connector_config_service.get_enabled(self._db, connector_id),
            ready=connector_config_cache.is_ready(connector_id),
            last_test_passed=passed,
            last_test_message=message,
            details=await self._settings_rows(),
        )
        if not summary.ready:
            return summary
        summary.details += [SummaryRow(label=k, value=v) for k, v in await self._engine.account_details()]
        await self._model_counts(summary)
        usage, note = await self._engine.usage_summary()
        summary.usage = ConnectorUsage(**usage) if usage else None
        summary.usage_note = note
        return summary
