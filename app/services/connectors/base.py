"""
The declarative shape a "Connector" is described by — the Connectors admin page (see
app/routers/connectors.py, app/templates/connectors.html) renders one generic
config form per registered ConnectorDefinition, looping over its own config_fields,
rather than hand-written per-connector HTML. Adding connector #2 later is one new
ConnectorDefinition + one line in registry.py — no template/JS change needed, as
long as its config is expressible as a flat set of text/password fields.

A Connector is deliberately a separate concept from engine registration itself (see
app.services.engines.registry): a connector describes *admin-facing config*, an
InferenceEngine describes *how calls are actually made*. RunPod has exactly one of
each, linked by `engine_name`.
"""

from dataclasses import dataclass, field
from typing import Literal

ConnectorFieldType = Literal["text", "password"]


@dataclass(frozen=True)
class ConnectorConfigField:
    """One form field in a connector's config — name is the key it's stored/read
    under (see app.services.connector_config_service). `secret=True` fields are
    encrypted at rest (app.services.secret_crypto) and never re-sent to the browser
    once saved (see connector_config_service.get_config_for_display)."""

    name: str
    label: str
    field_type: ConnectorFieldType
    required: bool = True
    secret: bool = False
    help_text: str = ""


@dataclass(frozen=True)
class ConnectorDefinition:
    """One entry in app.services.connectors.registry.CONNECTORS. `id` is a fixed,
    developer-chosen slug (never derived from user input — it's used directly in an
    AppSetting key, see connector_config_service.CONFIG_KEY_TEMPLATE) and `engine_name`
    is the EngineName the connector configures/enables (app.schemas.EngineName)."""

    id: str
    engine_name: str
    display_name: str
    description: str
    config_fields: tuple[ConnectorConfigField, ...] = field(default_factory=tuple)
