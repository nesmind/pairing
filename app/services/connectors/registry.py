"""Every Connector this app ships, listed explicitly (like engines.registry.registry, no import-order
dependence). Connector #2 is one ConnectorDefinition (see runpod_connector.py) plus one line here.
"""

from app.services.connectors.base import ConnectorDefinition
from app.services.connectors.cloudflare_connector import CLOUDFLARE_CONNECTOR
from app.services.connectors.runpod_connector import RUNPOD_CONNECTOR

CONNECTORS: list[ConnectorDefinition] = [RUNPOD_CONNECTOR, CLOUDFLARE_CONNECTOR]

_BY_ID = {connector.id: connector for connector in CONNECTORS}


def get_connector(connector_id: str) -> ConnectorDefinition:
    try:
        return _BY_ID[connector_id]
    except KeyError:
        raise KeyError(f"No connector registered under {connector_id!r} — registered: {sorted(_BY_ID)}") from None
