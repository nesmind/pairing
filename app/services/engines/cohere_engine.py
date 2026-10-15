"""CohereEngine: InferenceEngine adapter over cohere_client.PROVIDER (openai_compatible_engine)."""

from app.services import cohere_client
from app.services.cohere_client import CohereError
from app.services.engines.openai_compatible_engine import OpenAICompatibleEngine, cloud_capabilities


class CohereEngine(OpenAICompatibleEngine):
    name = "cohere"
    display_name = "Cohere"
    capabilities = cloud_capabilities(embeddings=True)
    error_types = (CohereError,)
    provider = cohere_client.PROVIDER
    usage_note = "Cohere has no usage API for this: see your calls and limits in the Cohere dashboard (Usage)."
