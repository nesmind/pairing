"""Reads per-file GGUF header metadata (architecture, display name) from a Hugging Face file with HTTP range
requests: a few MB, never the multi-GB weights. Split from huggingface_client (file-size rule): binary GGUF
parsing vs. JSON Hub API calls.

Needed because Hugging Face's repo-level `gguf` block describes only ONE file in a multi-file repo (its
primary), so other files (mmproj sidecar, embedding companion) would get that file's architecture/size.
Reading the file is the only way to know what it is before pulling: a 2MB range resolved "clip" for a
vision-projector and "phi2" for its sibling text model, matching what Matricxon found once pulled.
"""

import logging
import struct

import httpx

logger = logging.getLogger("llama_chat")

_RESOLVE_BASE = "https://huggingface.co"
_TIMEOUT = httpx.Timeout(15.0, connect=10.0)
_MAGIC = b"GGUF"

# The first range covers metadata even with a full vocab (51,200 tokens + 50,000 merges fit in ~1.8MB;
# general.architecture/general.name come first); the fallback is for unusual repos.
_FIRST_RANGE_BYTES = 2 * 1024 * 1024
_FALLBACK_RANGE_BYTES = 16 * 1024 * 1024

# GGUFValueType (llama.cpp/ggml spec) -> (struct format, byte size) for fixed-size scalars. STRING(8) and ARRAY(9) are
# variable-length, handled separately.
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
    """Raised when `data` ends before the key/value section does; probe_metadata retries with a larger range (not a
    parse failure)."""


def _read_string(data: bytes, offset: int) -> tuple[str, int]:
    if offset + 8 > len(data):
        raise _BufferExhausted
    (length,) = struct.unpack_from("<Q", data, offset)
    offset += 8
    if offset + length > len(data):
        raise _BufferExhausted
    return data[offset : offset + length].decode("utf-8", errors="replace"), offset + length


def _skip_value(data: bytes, offset: int, value_type: int) -> int:
    """Skips one metadata value: every key except general.architecture/general.name must still be skipped
    correctly, since GGUF entries have no byte length to jump by.
    """
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
            # No architecture this app pulls uses a nested array; refusing is safer than mis-parsing what follows.
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
    """None if `data` isn't GGUF (wrong magic; give up, don't retry). Raises _BufferExhausted if
    general.architecture/name/parameter_count may lie beyond what was fetched. Stops once all three are found or
    all pairs are seen. `parameter_count` is scalar-typed, read via the same `_SCALAR_TYPES` table as
    `_skip_value` (matching how the sibling matricxon repo builds `parameter_size`); left None if absent.
    """
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
    """general.architecture/name/parameter_count from `filename`'s GGUF header on Hugging Face via a couple of
    range requests. None on any failure (network, not GGUF, corrupt, metadata not found even at the larger
    range): best-effort, as ExtendedModelCatalog.add falls back to repo-level metadata and
    MatricxonDirectPuller treats a None parameter_count as "unknown".
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
                # Untrusted internet binary: anything unexpected degrades to "couldn't determine it", never failing
                # "Add".
                logger.warning("Failed to parse GGUF metadata for %s:%s", repo_id, filename, exc_info=True)
                return None
            return result
    return None
