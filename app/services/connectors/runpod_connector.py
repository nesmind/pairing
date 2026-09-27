"""RunPod Serverless — pAIring's first Connector. Talks to a RunPod Serverless
endpoint running the vLLM worker template's OpenAI-compatible API (see
app.services.runpod_client for the actual HTTP calls); no local process, no host
pool — an admin points this at an endpoint they've already deployed on RunPod."""

from app.services.connectors.base import ConnectorConfigField, ConnectorDefinition

RUNPOD_CONNECTOR = ConnectorDefinition(
    id="runpod",
    engine_name="runpod",
    display_name="RunPod Serverless",
    description=(
        "Route chat requests to a RunPod Serverless endpoint running the vLLM worker "
        "(OpenAI-compatible API). No local process — pAIring calls your endpoint's "
        "REST API directly. Does not support embeddings."
    ),
    config_fields=(
        ConnectorConfigField(
            name="endpoint_id",
            label="Endpoint ID",
            field_type="text",
            help_text="The RunPod Serverless endpoint ID (from your RunPod console's endpoint URL).",
        ),
        ConnectorConfigField(
            name="api_key",
            label="API key",
            field_type="password",
            secret=True,
            help_text="A RunPod API key with access to this endpoint.",
        ),
        ConnectorConfigField(
            name="model",
            label="Model name",
            field_type="text",
            help_text="The model name your RunPod vLLM endpoint serves (used if it can't be listed live).",
        ),
    ),
)
