"""Cohere (https://docs.cohere.com/docs/compatibility-api): Command chat models and Embed models through Cohere's
OpenAI-compatible API, usable with a free trial key. The compatibility API has no model list, so models come from
Cohere's own `GET /v1/models`, which says what each model is for."""

from app.services.openai_compatible import OpenAICompatibleProvider


class CohereError(Exception):
    """Cohere is unreachable, refused the call, or isn't configured and enabled on the Connectors page."""


class CohereProvider(OpenAICompatibleProvider):
    connector_id = "cohere"
    name = "Cohere"
    base_url = "https://api.cohere.ai/compatibility/v1"
    models_url = "https://api.cohere.com/v1/models?page_size=1000"
    key_help = "create one in the Cohere dashboard (dashboard.cohere.com > API keys)."
    error_type = CohereError

    def model_entries(self, body: dict) -> list[dict]:
        return body.get("models", [])

    def model_info(self, entry: dict) -> dict | None:
        """Chat models as "completion", Embed models as "embedding"; deprecated, rerank-only and other models are
        left out."""
        endpoints = set(entry.get("endpoints") or [])
        if entry.get("is_deprecated") or not entry.get("name"):
            return None
        if "chat" in endpoints:
            capability = "completion"
        elif "embed" in endpoints:
            capability = "embedding"
        else:
            return None
        info = super().model_info({"id": entry["name"], "context_length": entry.get("context_length")})
        info["capabilities"] = [capability]
        info["vendor"] = "Cohere"
        return info

    def embed_payload(self, text: str, model: str) -> dict:
        return {"model": model, "input": text, "encoding_format": "float"}

    def explain(self, exc: Exception, model: str) -> Exception | None:
        text = str(exc)
        if "(429)" in text:
            return CohereError(
                "Cohere's limit was reached: a trial key allows about 1,000 calls a month and a few a minute. "
                "Wait a little, or use a production key."
            )
        if "(404)" in text:
            return CohereError(f"Cohere has no model {model} for chat. Pick another model.")
        return None


PROVIDER = CohereProvider()
