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
