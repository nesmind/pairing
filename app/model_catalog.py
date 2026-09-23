"""
A curated list of chat models this app knows how to offer in Settings,
beyond whatever the user has already pulled into the ML engine manually. Each
entry is a *potential* model — app/routers/settings.py cross-references
this against what's actually installed (`ollama list`) and against this
machine's hardware (app/hardware.py) to decide, per entry, whether to
show "select", "pull", or a disabled row with a minimum-requirements
message.

The actual data lives in `default_models.json` at the repo root, not here — a hand-editable, git-tracked
file (never `.gitignore`d, ships with real defaults on a fresh clone) rather than Python literals, since this
is the one model list an admin has no in-app way to remove or edit (unlike the admin-managed "browse more
models" extended catalog — see app.services.extended_model_catalog_service — which has its own Add/Remove UI
backed by the database instead). `CuratedCatalogLoader.load` below reads and validates it; a malformed
hand-edit fails loudly at import time (a clear, per-field pydantic error) rather than silently dropping an
entry or falling back to an empty catalog.

Sizes/requirements in that file come from each model's public spec sheet at the time it was written — see
ROADMAP.md for how to keep this current as new sizes/models are released.

`min_ram_gb` is deliberately generous (roughly 1.25x the on-disk weight
size) to leave headroom for the context-window KV cache and the ML engine's own
runtime overhead — better to under-promise than to gate a model in as
"fits" and have it OOM or swap itself into uselessness.

Every entry also carries `architecture` (a GGUF `general.architecture` string) and `quantizations` (the real,
underlying GGML tensor type(s) the file actually uses — a plain quant like "Q4_0" is just itself, but a K-quant
"mix" label like "Q4_K_M" is llama.cpp's own convention for "mostly Q4_K, with certain higher-impact tensors
bumped to Q6_K", so it's recorded as the full `["Q4_K", "Q6_K"]` set, not the single mix label, which isn't
itself a real GGML type at all). Neither is Matricxon-specific by itself — both describe the file, not what can
run it.

Whether the sibling Matricxon engine (../matricxon, an independent from-scratch GGUF runtime, see
app.services.matricxon_client) can actually run a given entry is computed live from these two fields against
Matricxon's own real, current support surface — see app.services.matricxon_support_checker.MatricxonSupportChecker,
which calls Matricxon's own `GET /api/health` (added specifically for this — see ../matricxon/ROADMAP.md's
"Expose supported architectures/quantizations via a new API endpoint") rather than hand-copying its
ArchitectureRegistry/QuantStrategyRegistry contents into a second, driftable list here the way this file used
to. `vision=True` below no longer means an automatic, unconditional "not supported by Matricxon" — Matricxon
has real (if narrow: one architecture, single-image-view) vision support now, for a model with its mmproj file
actually pulled there too (see ../matricxon/ROADMAP.md's real LLaVA vision-support entry, 2026-09-21).

This is all independent of Ollama, which can run every entry this catalog offers regardless.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

_DEFAULT_MODELS_PATH = Path(__file__).resolve().parent.parent / "default_models.json"


@dataclass(frozen=True)
class CuratedModel:
    """One hand-curated, potentially-pullable model — real attributes (`entry.architecture`) instead of the
    plain dicts every entry used to be. Same fields default_models.json's own entries carry (minus `note`,
    which is metadata for a hand-editing admin only — see this module's own docstring); see this module's own
    docstring for what architecture/quantizations mean."""

    family: str
    vendor: str
    tag: str
    parameter_size: str
    context_length: int | None
    download_gb: float | None
    min_ram_gb: float
    locally_runnable: bool
    architecture: str
    quantizations: list[str] | None
    vision: bool = False
    unavailable_reason: str | None = None
    # Only ever set on an embedding_models entry — no chat-model entry has an embedding dimensionality.
    embedding_dim: int | None = None


class CuratedCatalog:
    """The admin-independent, hand-curated model list. Two disjoint collections, not one list with a type
    flag: an embedding-only entry can never leak into the chat-model picker
    (app.services.model_catalog_service) or vice versa."""

    def __init__(self, chat_models: list[CuratedModel], embedding_models: list[CuratedModel]) -> None:
        self._chat_models = chat_models
        self._embedding_models = embedding_models

    @property
    def chat_models(self) -> list[CuratedModel]:
        return self._chat_models

    @property
    def embedding_models(self) -> list[CuratedModel]:
        return self._embedding_models

    def find(self, tag: str) -> CuratedModel | None:
        """Looks up a chat-model catalog entry by its exact model tag."""
        return next((entry for entry in self._chat_models if entry.tag == tag), None)

    def find_embedding(self, tag: str) -> CuratedModel | None:
        """Looks up an embedding-catalog entry by its exact model tag — mirrors find() above, kept separate
        since embedding_models is a deliberately distinct collection (see its own docstring)."""
        return next((entry for entry in self._embedding_models if entry.tag == tag), None)


class _CuratedModelJSON(BaseModel):
    """Validates the shape of one default_models.json entry — a real boundary (a hand-edited external file,
    prone to typos) per CLAUDE.md, unlike CuratedModel above which stays a plain dataclass for everything
    already inside the app. `note` is accepted and then dropped (see CuratedCatalogLoader.load) rather than
    rejected as an unexpected field, since it's meant for whoever is editing the file, not for the app."""

    family: str
    vendor: str
    tag: str
    parameter_size: str
    context_length: int | None = None
    download_gb: float | None = None
    min_ram_gb: float
    locally_runnable: bool
    architecture: str
    quantizations: list[str] | None = None
    vision: bool = False
    unavailable_reason: str | None = None
    embedding_dim: int | None = None
    note: str | None = None


class CuratedCatalogLoader:
    @staticmethod
    def load(path: Path) -> CuratedCatalog:
        """Reads and validates `path` (default_models.json's own shape — {"chat_models": [...],
        "embedding_models": [...]}), raising straight through on anything malformed (bad JSON syntax, a
        missing required field, a wrong type) — a broken hand-edit should fail loudly at import time with a
        clear, specific error pointing at the exact bad field, not silently drop an entry or fall back to an
        empty catalog that would make every model in Settings quietly disappear."""
        raw = json.loads(path.read_text())
        return CuratedCatalog(
            chat_models=[CuratedCatalogLoader._to_model(entry) for entry in raw["chat_models"]],
            embedding_models=[CuratedCatalogLoader._to_model(entry) for entry in raw["embedding_models"]],
        )

    @staticmethod
    def _to_model(entry: dict) -> CuratedModel:
        validated = _CuratedModelJSON(**entry)
        return CuratedModel(**validated.model_dump(exclude={"note"}))


CATALOG = CuratedCatalogLoader.load(_DEFAULT_MODELS_PATH)
