"""Every Connector this app ships, listed explicitly (like engines.registry.registry, no import-order
dependence). Connector #2 is one ConnectorDefinition (see runpod_connector.py) plus one line here.
"""

from app.services.connectors.base import ConnectorDefinition
from app.services.connectors.cloudflare_connector import CLOUDFLARE_CONNECTOR
from app.services.connectors.cohere_connector import COHERE_CONNECTOR
from app.services.connectors.digitalocean_connector import DIGITALOCEAN_CONNECTOR
from app.services.connectors.gemini_connector import GEMINI_CONNECTOR
from app.services.connectors.groq_connector import GROQ_CONNECTOR
from app.services.connectors.nvidia_connector import NVIDIA_CONNECTOR
from app.services.connectors.openrouter_connector import OPENROUTER_CONNECTOR
from app.services.connectors.runpod_connector import RUNPOD_CONNECTOR

CONNECTORS: list[ConnectorDefinition] = [
    CLOUDFLARE_CONNECTOR,
    DIGITALOCEAN_CONNECTOR,
    OPENROUTER_CONNECTOR,
    GEMINI_CONNECTOR,
    GROQ_CONNECTOR,
    NVIDIA_CONNECTOR,
    COHERE_CONNECTOR,
    RUNPOD_CONNECTOR,
]

_BY_ID = {connector.id: connector for connector in CONNECTORS}


def get_connector(connector_id: str) -> ConnectorDefinition:
    try:
        return _BY_ID[connector_id]
    except KeyError:
        raise KeyError(f"No connector registered under {connector_id!r} — registered: {sorted(_BY_ID)}") from None
