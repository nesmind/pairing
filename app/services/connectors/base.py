"""The declarative shape of a "Connector". The Connectors admin page (routers/connectors.py, connectors.html)
renders one generic config form per ConnectorDefinition from its config_fields, so connector #2 is one
ConnectorDefinition plus one line in registry.py, with no template/JS change, if its config is flat text/password
fields.

A Connector is separate from engine registration (engines.registry): a connector is *admin-facing config*,
an InferenceEngine is *how calls are made*. RunPod has one of each, linked by `engine_name`.
"""

from dataclasses import dataclass, field
from typing import Literal

ConnectorFieldType = Literal["text", "password"]


@dataclass(frozen=True)
class ConnectorConfigField:
    """One config form field; name is its storage/read key (connector_config_service). `secret=True` fields are
    encrypted at rest (secret_crypto) and never re-sent to the browser (get_config_for_display).
    """

    name: str
    label: str
    field_type: ConnectorFieldType
    required: bool = True
    secret: bool = False
    help_text: str = ""


@dataclass(frozen=True)
class ConnectorDefinition:
    """One entry of connectors.registry.CONNECTORS. `id` is a developer-chosen slug, never from user input (used
    in an AppSetting key, connector_config_service.CONFIG_KEY_TEMPLATE); `engine_name` is the EngineName it configures
    (app.schemas.EngineName).
    """

    id: str
    engine_name: str
    display_name: str
    description: str
    config_fields: tuple[ConnectorConfigField, ...] = field(default_factory=tuple)
