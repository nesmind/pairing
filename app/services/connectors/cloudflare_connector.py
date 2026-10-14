"""Cloudflare Workers AI: models hosted by Cloudflare, reached through its OpenAI-compatible API
(calls: cloudflare_client). No endpoint to deploy; chat and embeddings both work."""

from app.services.connectors.base import ConnectorConfigField, ConnectorDefinition

CLOUDFLARE_CONNECTOR = ConnectorDefinition(
    id="cloudflare",
    engine_name="cloudflare",
    display_name="Cloudflare Workers AI",
    description=(
        "Use the models Cloudflare hosts on its global GPU network: dozens of chat models (Gemma, Llama, Qwen, "
        "Mistral, DeepSeek, GPT-OSS and more) plus embedding models for the knowledge base. Nothing to deploy — "
        "pAIring calls Cloudflare's API directly. Text only for now (no images or tools)."
    ),
    config_fields=(
        ConnectorConfigField(
            name="account_id",
            label="Account ID",
            field_type="text",
            help_text="Your Cloudflare account ID (dashboard > Workers AI, or any domain's Overview page).",
        ),
        ConnectorConfigField(
            name="api_token",
            label="API token",
            field_type="password",
            secret=True,
            help_text="An API token with the Workers AI permission (dashboard > My Profile > API Tokens).",
        ),
    ),
)
