"""Declarative descriptions of pluggable third-party engines (base.py: what a Connector is; registry.py: the list)."""

from app.services.connectors.base import ConnectorConfigField, ConnectorDefinition
from app.services.connectors.registry import CONNECTORS, get_connector

__all__ = ["CONNECTORS", "ConnectorConfigField", "ConnectorDefinition", "get_connector"]
