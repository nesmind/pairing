"""
Reads real per-file GGUF header metadata (architecture, display name) directly from a Hugging Face-hosted file
via HTTP range requests — a few MB at most, never the full (often multi-GB) weights. Split out of
app.services.huggingface_client (file-size rule) since this is byte-level GGUF binary parsing, a distinct
concern from that module's plain JSON Hub API calls.

Exists because Hugging Face's own repo-level `gguf` metadata block (HuggingFaceCatalogSearch.repo_files)
describes only ONE file per multi-file repo (confirmed live: HF picks whichever file it treats as primary) —
every other file in that same repo (a vision-projector/mmproj sidecar, an embedding-only companion, ...) gets
misattributed that file's architecture/size if a caller just trusts the repo-level block uncritically for it.
Reading the real file directly is the only way to know what a *specific* file actually is before pulling it —
confirmed live against two real files in the same repo: a 2MB range read correctly resolved "clip" for a
vision-projector sidecar and "phi2" for its sibling text model, matching what Matricxon itself determined
independently once each was actually pulled.
"""

import logging
import struct

import httpx

logger = logging.getLogger("llama_chat")

_RESOLVE_BASE = "https://huggingface.co"
_TIMEOUT = httpx.Timeout(15.0, connect=10.0)
_MAGIC = b"GGUF"

# First-pass range covers real metadata comfortably even with a full tokenizer vocab (confirmed live: a real
# 51,200-token vocab + 50,000 BPE merges still fit in ~1.8MB, and general.architecture/general.name are written
# before any of that on every real file checked) — the fallback only matters for a genuinely unusual repo.
_FIRST_RANGE_BYTES = 2 * 1024 * 1024
_FALLBACK_RANGE_BYTES = 16 * 1024 * 1024

# GGUFValueType (the real llama.cpp/ggml GGUF spec) -> (struct format, byte size) for every fixed-size scalar
# type. STRING(8) and ARRAY(9) are handled separately below (variable-length).
_SCALAR_TYPES = {
    0: ("<B", 1),
    1: ("<b", 1),
    2: ("<H", 2),
    3: ("<h", 2),
    4: ("<I", 4),
    5: ("<i", 4),
    6: ("<f", 4),
    7: ("<?", 1),
    10: ("<Q", 8),
    11: ("<q", 8),
    12: ("<d", 8),
}
_STRING_TYPE = 8
_ARRAY_TYPE = 9


class _BufferExhausted(Exception):
    """Raised internally when `data` runs out before the key/value section finishes parsing — signals
    probe_metadata to retry with a larger range, not a real parse failure."""


def _read_string(data: bytes, offset: int) -> tuple[str, int]:
    if offset + 8 > len(data):
        raise _BufferExhausted
    (length,) = struct.unpack_from("<Q", data, offset)
    offset += 8
    if offset + length > len(data):
        raise _BufferExhausted
    return data[offset : offset + length].decode("utf-8", errors="replace"), offset + length


def _skip_value(data: bytes, offset: int, value_type: int) -> int:
    """Advances past one metadata value without keeping it — every value this probe doesn't care about (every
    key except general.architecture/general.name) still has to be skipped correctly to reach the next key/value
    pair, since GGUF's own header has no per-entry byte length of its own to jump by."""
    if value_type == _STRING_TYPE:
        _, offset = _read_string(data, offset)
        return offset
    if value_type == _ARRAY_TYPE:
        if offset + 12 > len(data):
            raise _BufferExhausted
        (element_type,) = struct.unpack_from("<I", data, offset)
        (length,) = struct.unpack_from("<Q", data, offset + 4)
        offset += 12
        if element_type == _STRING_TYPE:
            for _ in range(length):
                _, offset = _read_string(data, offset)
            return offset
        if element_type == _ARRAY_TYPE:
            # No real architecture this app pulls defines a nested array — refusing outright is safer than
            # silently mis-parsing everything after it.
            raise _BufferExhausted
        _, size = _SCALAR_TYPES[element_type]
        end = offset + size * length
        if end > len(data):
            raise _BufferExhausted
        return end
    _, size = _SCALAR_TYPES[value_type]
    if offset + size > len(data):
        raise _BufferExhausted
    return offset + size


def _parse_metadata(data: bytes) -> dict[str, str | int | None] | None:
    """None if `data` isn't a GGUF file at all (wrong magic) — a real "give up," not a retry-with-more-bytes
    case. Raises _BufferExhausted if general.architecture/general.name/general.parameter_count might still be
    further in than what was fetched. Stops as soon as all three are found, or once every key/value pair in
    the header has been seen. `parameter_count` (the same key
    app.services.extended_model_catalog_service's sibling matricxon repo reads via `metadata.get("general.
    parameter_count")` to build its own sidecar's `parameter_size` display string) is scalar-typed, not a
    string, so it's read via the same `_SCALAR_TYPES` table `_skip_value` already uses to skip past values
    this probe doesn't care about — real files don't always carry it, so it's left None rather than guessed
    at, same as architecture/name when a file genuinely doesn't have them."""
    if data[:4] != _MAGIC:
        return None
    if len(data) < 24:
        raise _BufferExhausted
    (kv_count,) = struct.unpack_from("<Q", data, 16)
    offset = 24
    found: dict[str, str | int | None] = {
        "architecture": None,
        "name": None,
        "parameter_count": None,
    }
    for _ in range(kv_count):
        key, offset = _read_string(data, offset)
        if offset + 4 > len(data):
            raise _BufferExhausted
        (value_type,) = struct.unpack_from("<I", data, offset)
        offset += 4
        if key in ("general.architecture", "general.name") and value_type == _STRING_TYPE:
            value, offset = _read_string(data, offset)
            found[key.removeprefix("general.")] = value
        elif key == "general.parameter_count" and value_type in _SCALAR_TYPES:
            fmt, size = _SCALAR_TYPES[value_type]
            if offset + size > len(data):
                raise _BufferExhausted
            (found["parameter_count"],) = struct.unpack_from(fmt, data, offset)
            offset += size
        else:
            offset = _skip_value(data, offset, value_type)
        if all(value is not None for value in found.values()):
            break
    return found


async def probe_metadata(repo_id: str, filename: str, proxy_url: str | None) -> dict[str, str | int | None] | None:
    """Real general.architecture/general.name/general.parameter_count read directly from `filename`'s own
    GGUF header on Hugging Face — via a couple of HTTP range requests, never the full file. Returns None on
    any failure (network error, not actually a GGUF file, corrupt/unexpected binary content, or genuinely
    can't find the metadata even at the larger fallback range) — always best-effort:
    app.services.extended_model_catalog_service.ExtendedModelCatalog.add falls back to Hugging Face's own
    coarser repo-level metadata instead of failing the whole "Add" action over this, and
    app.services.matricxon_direct_puller.MatricxonDirectPuller treats a None parameter_count the same way
    Matricxon's own sidecar-building code does — "unknown", not a hard failure.
    """
    url = f"{_RESOLVE_BASE}/{repo_id}/resolve/main/{filename}"
    async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True, proxy=proxy_url) as client:
        for range_bytes in (_FIRST_RANGE_BYTES, _FALLBACK_RANGE_BYTES):
            try:
                resp = await client.get(url, headers={"Range": f"bytes=0-{range_bytes - 1}"})
            except httpx.HTTPError:
                return None
            if resp.status_code not in (200, 206):
                return None
            try:
                result = _parse_metadata(resp.content)
            except _BufferExhausted:
                continue
            except Exception:
                # Untrusted binary content from the open internet — anything genuinely unexpected here must
                # degrade to "couldn't determine it," never take the whole "Add" action down with it.
                logger.warning("Failed to parse GGUF metadata for %s:%s", repo_id, filename, exc_info=True)
                return None
            return result
    return None
