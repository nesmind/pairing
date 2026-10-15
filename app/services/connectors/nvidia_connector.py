"""NVIDIA NIM (build.nvidia.com): 100+ hosted models with free developer credits (calls: nvidia_client)."""

from app.services.connectors.base import ConnectorConfigField, ConnectorDefinition

NVIDIA_CONNECTOR = ConnectorDefinition(
    id="nvidia",
    engine_name="nvidia",
    display_name="NVIDIA NIM",
    description=(
        "More than 100 hosted models from NVIDIA's API catalog (Llama, Nemotron, Mistral, DeepSeek, Qwen and more), "
        "plus embedding models for the knowledge base. A free developer account comes with credits and a rate limit; "
        "NVIDIA's terms aim them at trying models, not production use. Text only for now (no images or tools)."
    ),
    config_fields=(
        ConnectorConfigField(
            name="api_key",
            label="API key",
            field_type="password",
            secret=True,
            help_text="On build.nvidia.com, open any model and press Get API Key (it starts with nvapi-).",
        ),
    ),
)
