"""GET/PUT shapes for the Connectors page (routers/connectors.py, services.connectors)."""

from typing import Literal

from pydantic import BaseModel

from app.schemas.common import EngineName


class ConnectorConfigFieldOut(BaseModel):
    """Mirrors connectors.base.ConnectorConfigField: one form field, never a saved value (see ConnectorOut.values)."""

    name: str
    label: str
    field_type: Literal["text", "password"]
    required: bool
    secret: bool
    help_text: str


class ConnectorOut(BaseModel):
    """One entry of GET /api/connectors: static definition plus admin state. `values` comes from
    connector_config_service.get_config_for_display; secrets appear as `has_<field>: bool`.
    """

    id: str
    engine_name: EngineName
    display_name: str
    description: str
    config_fields: list[ConnectorConfigFieldOut]
    values: dict[str, str | bool]
    configured: bool
    enabled: bool
    ready: bool
    # Result of the last "Test connection" on the saved config. False with no message = never tested or config
    # changed since (connector_config_service.set_config resets on save).
    last_test_passed: bool = False
    last_test_message: str | None = None


class ConnectorsResponse(BaseModel):
    connectors: list[ConnectorOut]


class ConnectorConfigUpdate(BaseModel):
    """PUT /api/connectors/{id}/config body: field name to value. A missing/blank secret keeps the saved value."""

    values: dict[str, str]


class ConnectorEnabledUpdate(BaseModel):
    enabled: bool


class SummaryRow(BaseModel):
    label: str
    value: str


class ModelUsage(BaseModel):
    model: str
    requests: int
    units: float


class ConnectorUsage(BaseModel):
    """Usage the provider reports (cloudflare_client.usage_summary). `units` are its billing units, e.g. neurons."""

    period: str
    units_label: str
    requests: int
    units: float
    input_tokens: int
    output_tokens: int
    today_requests: int
    today_units: float
    top_models: list[ModelUsage]


class ConnectorSummary(BaseModel):
    """GET /api/connectors/{id}/summary: what the External servers tab shows for a connector engine."""

    id: str
    display_name: str
    enabled: bool
    ready: bool
    last_test_passed: bool
    last_test_message: str | None = None
    # Non-secret settings plus what the provider reports (e.g. token status).
    details: list[SummaryRow]
    chat_models: int | None = None
    embedding_models: int | None = None
    models_error: str | None = None
    usage: ConnectorUsage | None = None
    # Why usage is missing, when it is (e.g. a token permission).
    usage_note: str | None = None
