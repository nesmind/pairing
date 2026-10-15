"""Readable names and vendors for cloud model ids, which carry no Hugging Face metadata:
"@cf/meta/llama-3.2-1b-instruct" (Cloudflare), "meta-llama/Meta-Llama-3.1-8B-Instruct" or
"anthropic-claude-3.5-sonnet" (DigitalOcean)."""

import re

# Org (or leading name word) -> how the vendor reads; others are title-cased.
VENDORS = {
    "meta": "Meta",
    "meta-llama": "Meta",
    "llama": "Meta",
    "google": "Google",
    "gemma": "Google",
    "qwen": "Qwen",
    "alibaba": "Qwen",
    "mistral": "Mistral AI",
    "mistralai": "Mistral AI",
    "deepseek": "DeepSeek",
    "deepseek-ai": "DeepSeek",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "baai": "BAAI",
    "moonshotai": "Moonshot AI",
    "zai-org": "Z.ai",
    "ibm-granite": "IBM",
    "nvidia": "NVIDIA",
    "aisingapore": "AI Singapore",
    "glm": "Z.ai",
    "kimi": "Moonshot AI",
    "nemotron": "NVIDIA",
    "minimax": "MiniMax",
    "mimo": "Xiaomi",
    "arcee": "Arcee AI",
    "router": "DigitalOcean",
    "bge": "BAAI",
    "gte": "Alibaba",
    "e5": "Microsoft",
    "command": "Cohere",
    "aya": "Cohere",
    "c4ai": "Cohere",
}
# Leading words that only name the vendor, dropped from the display name ("openai-gpt-4o" -> "GPT 4o").
_VENDOR_ONLY_WORDS = {"openai", "anthropic", "alibaba"}
_UPPER_WORDS = {"gpt", "oss", "glm", "bge", "fp8", "it", "r1", "en", "vl", "hf", "gte", "e5", "lm", "qa", "tts"}
_WORD_FIXES = {"deepseek": "DeepSeek", "minimax": "MiniMax", "mimo": "MiMo"}
_SIZE_WORD = re.compile(r"[a-z]?\d+(\.\d+)?[bm]", re.IGNORECASE)  # 8b, 0.6b, a12b
_SPLIT = re.compile(r"[-_:\s]+")


class ReadableModelName:
    """Readable name and vendor from a cloud model id; `fallback_vendor` when neither org nor name tells."""

    def __init__(self, model_id: str, fallback_vendor: str = "") -> None:
        parts = [p for p in model_id.split("/") if p and not p.startswith("@")]
        self._org = parts[-2].lower() if len(parts) >= 2 else ""
        self._model = parts[-1] if parts else model_id
        self._fallback = fallback_vendor

    def _first_word(self) -> str:
        word = _SPLIT.split(self._model)[0].lower()
        return re.sub(r"[\d.]+$", "", word)  # "llama3.3" -> "llama"

    @property
    def vendor(self) -> str:
        if self._org:
            return VENDORS.get(self._org, self._org.replace("-", " ").title())
        return VENDORS.get(self._first_word(), self._fallback)

    @property
    def display_name(self) -> str:
        words = _SPLIT.split(self._model)
        if not self._org and len(words) > 1 and words[0].lower() in _VENDOR_ONLY_WORDS:
            words = words[1:]
        readable = []
        for word in words:
            if word.lower() in _WORD_FIXES:
                readable.append(_WORD_FIXES[word.lower()])
            elif _SIZE_WORD.fullmatch(word) or word.lower() in _UPPER_WORDS:
                readable.append(word.upper())
            else:
                readable.append(word[:1].upper() + word[1:])
        return " ".join(readable)
