"""Groq: fast hosted open models with a free tier (calls: groq_client). Chat only, no embeddings."""

from app.services.connectors.base import ConnectorConfigField, ConnectorDefinition

GROQ_CONNECTOR = ConnectorDefinition(
    id="groq",
    engine_name="groq",
    display_name="Groq",
    description=(
        "Very fast replies from hosted open models (Llama, Qwen, GPT-OSS and more), with a free tier that needs no "
        "card. Free use has rate limits per model. Chat only: no embedding models, so the knowledge base can't be "
        "used while Groq is the active engine. Text only for now (no images or tools)."
    ),
    config_fields=(
        ConnectorConfigField(
            name="api_key",
            label="API key",
            field_type="password",
            secret=True,
            help_text="Create one at console.groq.com under API Keys.",
        ),
    ),
)
