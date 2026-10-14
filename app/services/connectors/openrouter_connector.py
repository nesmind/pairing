"""OpenRouter: one key for hundreds of models from many providers (calls: openrouter_client). Lists only the free
models unless set to "all"; chat and embeddings both work."""

from app.services.connectors.base import ConnectorConfigField, ConnectorDefinition

OPENROUTER_CONNECTOR = ConnectorDefinition(
    id="openrouter",
    engine_name="openrouter",
    display_name="OpenRouter",
    description=(
        "One key for hundreds of models from many providers. Lists only the free models by default, so you can try it "
        "at no cost. Note: free models may log or train on your prompts. Text only for now (no images or tools)."
    ),
    config_fields=(
        ConnectorConfigField(
            name="api_key",
            label="API key",
            field_type="password",
            secret=True,
            help_text="Create one on openrouter.ai under Settings > Keys.",
        ),
        ConnectorConfigField(
            name="models",
            label="Models to list",
            field_type="text",
            required=False,
            help_text='"free" (default) for the free models only, or "all" for every model, which costs credits.',
        ),
    ),
)
