"""GET/PUT shapes for the Connectors page (see app/routers/connectors.py and
app.services.connectors for what a Connector actually is)."""

from typing import Literal

from pydantic import BaseModel

from app.schemas.common import EngineName


class ConnectorConfigFieldOut(BaseModel):
    """Mirrors app.services.connectors.base.ConnectorConfigField — describes one
    form field, never a saved value (see ConnectorOut.values for that)."""

    name: str
    label: str
    field_type: Literal["text", "password"]
    required: bool
    secret: bool
    help_text: str


class ConnectorOut(BaseModel):
    """One entry in GET /api/connectors: a connector's static definition plus its
    live admin-configured state. `values` is whatever
    app.services.connector_config_service.get_config_for_display returns — a secret
    field appears as `has_<field>: bool`, never its real value."""

    id: str
    engine_name: EngineName
    display_name: str
    description: str
    config_fields: list[ConnectorConfigFieldOut]
    values: dict[str, str | bool]
    configured: bool
    enabled: bool
    ready: bool
    # The last "Test connection" action's result against the *currently saved* config — False with no message
    # means never tested, or the config has changed since (see connector_config_service.set_config's
    # reset-on-save behavior).
    last_test_passed: bool = False
    last_test_message: str | None = None


class ConnectorsResponse(BaseModel):
    connectors: list[ConnectorOut]


class ConnectorConfigUpdate(BaseModel):
    """PUT /api/connectors/{id}/config body — field name to submitted value. A
    missing/blank secret field means "keep the existing saved value" (see
    connector_config_service.set_config)."""

    values: dict[str, str]


class ConnectorEnabledUpdate(BaseModel):
    enabled: bool
