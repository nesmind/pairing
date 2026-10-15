"""Cohere: Command chat models and Embed models with a free trial key (calls: cohere_client)."""

from app.services.connectors.base import ConnectorConfigField, ConnectorDefinition

COHERE_CONNECTOR = ConnectorDefinition(
    id="cohere",
    engine_name="cohere",
    display_name="Cohere",
    description=(
        "Cohere's Command chat models (strong at answering from documents) and Embed models for the knowledge base, "
        "many languages included. A free trial key allows about 1,000 calls a month and is meant for trying, not "
        "production use. Text only for now (no images or tools)."
    ),
    config_fields=(
        ConnectorConfigField(
            name="api_key",
            label="API key",
            field_type="password",
            secret=True,
            help_text="Create one in the Cohere dashboard (dashboard.cohere.com > API keys). A trial key is free.",
        ),
    ),
)
