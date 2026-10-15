"""NVIDIA NIM on build.nvidia.com (https://integrate.api.nvidia.com/v1): 100+ hosted models through an
OpenAI-compatible API, with free credits for a developer account. Lists text chat and embedding models only."""

import re

import httpx

from app.services.openai_compatible import OpenAICompatibleProvider

_EMBEDDING_NAME = re.compile(r"embed|(^|[-_/])e5([-_]|$)|bge|arctic-embed", re.IGNORECASE)
# Listed with the chat models but not text chat: safety, reranking, speech, vision, image/video and science models.
_NOT_TEXT_NAME = re.compile(
    r"rerank|reward|guard|safety|jailbreak|topic-control|pii|parse|ocr|detect|clip|tts|asr|whisper|riva|canary|"
    r"parakeet|magpie|audio|vision|vlm|[-_]vl([-_]|$)|neva|vila|kosmos|paligemma|deplot|cosmos|video|diffusion|"
    r"flux|sdxl|stable|fourcastnet|esm|genmol|molmim|diffdock|proteinmpnn|openfold|alphafold|evo2|msa|cuopt",
    re.IGNORECASE,
)
# Some NIM models default to a short reply when no length is sent.
_DEFAULT_MAX_TOKENS = 4096
_MIN_MAX_TOKENS = 1024  # floor for short requests (titles): reasoning models think first


class NvidiaError(Exception):
    """NVIDIA NIM is unreachable, refused the call, or isn't configured and enabled on the Connectors page."""


class NvidiaProvider(OpenAICompatibleProvider):
    connector_id = "nvidia"
    name = "NVIDIA NIM"
    base_url = "https://integrate.api.nvidia.com/v1"
    key_help = "create one on build.nvidia.com (any model's page > Get API Key)."
    error_type = NvidiaError

    def model_info(self, entry: dict) -> dict | None:
        info = super().model_info(entry)
        if info is None:
            return None
        if _EMBEDDING_NAME.search(info["name"]):
            info["capabilities"] = ["embedding"]
        elif _NOT_TEXT_NAME.search(info["name"]):
            return None
        return info

    def chat_payload(self, model: str, messages: list[dict], params: dict, *, stream: bool) -> dict:
        payload = super().chat_payload(model, messages, params, stream=stream)
        limit = params.get("num_predict", -1)
        payload["max_tokens"] = _DEFAULT_MAX_TOKENS if limit in (-1, None) else max(limit, _MIN_MAX_TOKENS)
        return payload

    def embed_payload(self, text: str, model: str) -> dict:
        # Retrieval models (NV-Embed-QA, E5...) need a mode; "query" for both documents and questions keeps the
        # vectors comparable (embed has no document/question flag).
        return {"model": model, "input": text, "input_type": "query", "encoding_format": "float"}

    async def embed(self, text: str, model: str) -> list[float]:
        try:
            return await self.post_embedding(self.embed_payload(text, model))
        except NvidiaError as exc:
            cause = exc.__cause__
            if not (isinstance(cause, httpx.HTTPStatusError) and cause.response.status_code in (400, 422)):
                raise
        return await self.post_embedding({"model": model, "input": text})  # a model that takes no mode

    def explain(self, exc: Exception, model: str) -> Exception | None:
        text = str(exc)
        if "(402)" in text or "(429)" in text:
            return NvidiaError(
                "NVIDIA refused the request: your free credits or rate limit (about 40 requests a minute) ran out. "
                "Check your credits on build.nvidia.com, or wait a minute."
            )
        if "(404)" in text:
            return NvidiaError(f"NVIDIA has no hosted endpoint for {model} on this key. Pick another model.")
        return None


PROVIDER = NvidiaProvider()
