"""Shared by the engines that speak the OpenAI chat API: RunPod's vLLM worker (runpod_client) and Cloudflare
Workers AI (cloudflare_client). Reads a streamed /chat/completions reply as text chunks."""

import json
from collections.abc import AsyncGenerator

import httpx


class StreamErrorEvent(Exception):
    """The provider reported an error inside the stream (e.g. OpenRouter's mid-stream rate limit), not as HTTP."""


class OpenAIChatStream:
    """Streams one chat completion; any failure becomes `error_type`, so callers keep their own error class."""

    def __init__(self, provider: str, error_type: type[Exception]) -> None:
        self._provider = provider
        self._error_type = error_type

    @staticmethod
    def content_of(line: str) -> str | None:
        """The text in one SSE line; "" for a line without text, None at the end of the stream."""
        if not line.startswith("data:"):
            return ""
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            return None
        event = json.loads(data)
        if event.get("error"):
            error = event["error"]
            raise StreamErrorEvent(error.get("message", str(error)) if isinstance(error, dict) else str(error))
        choices = event.get("choices") or [{}]
        return (choices[0].get("delta") or {}).get("content") or ""

    async def stream(
        self, url: str, payload: dict, headers: dict[str, str], timeout: httpx.Timeout
    ) -> AsyncGenerator[str, None]:
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                async with client.stream("POST", url, json=payload, headers=headers) as resp:
                    try:
                        resp.raise_for_status()
                    except httpx.HTTPStatusError as exc:
                        # Read the reason now: the body is gone once the stream closes.
                        detail = (await resp.aread()).decode(errors="replace")[:300]
                        raise self._error_type(
                            f"{self._provider} chat request failed ({resp.status_code}): {detail}"
                        ) from exc
                    async for line in resp.aiter_lines():
                        try:
                            content = self.content_of(line)
                        except StreamErrorEvent as exc:
                            raise self._error_type(f"{self._provider} stopped the reply: {exc}") from exc
                        if content is None:
                            break
                        if content:
                            yield content
            except httpx.HTTPError as exc:
                raise self._error_type(f"{self._provider} chat request failed: {exc}") from exc
