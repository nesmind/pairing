"""Google Gemini API: Google's Gemini and Gemma models (calls: gemini_client). One API key from Google AI Studio;
chat and embeddings both work."""

from app.services.connectors.base import ConnectorConfigField, ConnectorDefinition

GEMINI_CONNECTOR = ConnectorDefinition(
    id="gemini",
    engine_name="gemini",
    display_name="Google Gemini",
    description=(
        "Google's Gemini and Gemma models, plus Gemini embeddings for the knowledge base. The free tier needs no card, "
        "but Google uses free-tier prompts to improve its products and people may review them; in the EEA, UK and "
        "Switzerland only the paid tier may be used. Text only for now (no images or tools)."
    ),
    config_fields=(
        ConnectorConfigField(
            name="api_key",
            label="API key",
            field_type="password",
            secret=True,
            help_text="Create one in Google AI Studio (aistudio.google.com > Get API key).",
        ),
    ),
)
