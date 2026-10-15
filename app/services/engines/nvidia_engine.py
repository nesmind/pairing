"""NvidiaEngine: InferenceEngine adapter over nvidia_client.PROVIDER (openai_compatible_engine)."""

from app.services import nvidia_client
from app.services.engines.openai_compatible_engine import OpenAICompatibleEngine, cloud_capabilities
from app.services.nvidia_client import NvidiaError


class NvidiaEngine(OpenAICompatibleEngine):
    name = "nvidia"
    display_name = "NVIDIA NIM"
    capabilities = cloud_capabilities(embeddings=True)
    error_types = (NvidiaError,)
    provider = nvidia_client.PROVIDER
    usage_note = "NVIDIA has no usage API for this: see your remaining credits on build.nvidia.com (your account)."
