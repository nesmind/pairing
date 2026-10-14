"""DigitalOcean Serverless Inference: models DigitalOcean hosts, reached through its OpenAI-compatible
API (calls: digitalocean_client). One model access key; chat and embeddings both work."""

from app.services.connectors.base import ConnectorConfigField, ConnectorDefinition

DIGITALOCEAN_CONNECTOR = ConnectorDefinition(
    id="digitalocean",
    engine_name="digitalocean",
    display_name="DigitalOcean Serverless Inference",
    description=(
        "Use the models DigitalOcean hosts for Serverless Inference: open models such as Llama, DeepSeek, "
        "Mistral and Qwen, and commercial ones such as OpenAI and Anthropic where your account offers them, plus "
        "embedding models for the knowledge base. Nothing to deploy. Text only for now (no images or tools)."
    ),
    config_fields=(
        ConnectorConfigField(
            name="api_key",
            label="Model access key",
            field_type="password",
            secret=True,
            help_text="Control panel: Inference Engine > Serverless Inference > Model Access Keys.",
        ),
    ),
)
