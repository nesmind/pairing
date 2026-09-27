"""Declarative descriptions of pluggable third-party engines — see base.py for what a
Connector actually is and registry.py for the list every one of them is registered
in."""

from app.services.connectors.base import ConnectorConfigField, ConnectorDefinition
from app.services.connectors.registry import CONNECTORS, get_connector

__all__ = ["CONNECTORS", "ConnectorConfigField", "ConnectorDefinition", "get_connector"]
