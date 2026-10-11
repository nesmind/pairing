"""Tests for services/gguf_probe.py. httpx.AsyncClient is replaced by a fake recording calls and returning canned
responses (as tests/test_huggingface_client.py, tests/test_ollama_client.py). GGUF byte buffers are built by hand
(`_build_gguf` and helpers), not fixture files, matching the module's byte-level scope.
"""

import struct

import httpx
import pytest

from app.services import gguf_probe as probe

_STRING_TYPE = 8
_ARRAY_TYPE = 9
_BOOL_TYPE = 7
_UINT32_TYPE = 4
_UINT64_TYPE = 10


def _gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _kv_string(key: str, value: str) -> bytes:
    return _gguf_string(key) + struct.pack("<I", _STRING_TYPE) + _gguf_string(value)


def _kv_bool(key: str, value: bool) -> bytes:
    return _gguf_string(key) + struct.pack("<I", _BOOL_TYPE) + struct.pack("<?", value)


def _kv_u64(key: str, value: int) -> bytes:
    return _gguf_string(key) + struct.pack("<I", _UINT64_TYPE) + struct.pack("<Q", value)


def _kv_u32_array(key: str, values: list[int]) -> bytes:
    header = _gguf_string(key) + struct.pack("<I", _ARRAY_TYPE)
    header += struct.pack("<I", _UINT32_TYPE) + struct.pack("<Q", len(values))
    return header + b"".join(struct.pack("<I", v) for v in values)


def _kv_string_array(key: str, values: list[str]) -> bytes:
    header = _gguf_string(key) + struct.pack("<I", _ARRAY_TYPE)
    header += struct.pack("<I", _STRING_TYPE) + struct.pack("<Q", len(values))
    return header + b"".join(_gguf_string(v) for v in values)


def _build_gguf(kv_blobs: list[bytes], tensor_count: int = 0) -> bytes:
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", tensor_count) + struct.pack("<Q", len(kv_blobs))
    return header + b"".join(kv_blobs)


# _parse_metadata (pure, no network)


def test_parse_metadata_returns_none_for_wrong_magic():
    assert probe._parse_metadata(b"NOPE" + b"\x00" * 20) is None


def test_parse_metadata_raises_buffer_exhausted_for_a_too_short_header():
    with pytest.raises(probe._BufferExhausted):
        probe._parse_metadata(b"GGUF\x03\x00\x00\x00")


def test_parse_metadata_extracts_architecture_and_name():
    data = _build_gguf([_kv_string("general.architecture", "phi2"), _kv_string("general.name", "moondream2")])
    assert probe._parse_metadata(data) == {"architecture": "phi2", "name": "moondream2", "parameter_count": None}


def test_parse_metadata_extracts_parameter_count():
    data = _build_gguf([_kv_string("general.architecture", "phi2"), _kv_u64("general.parameter_count", 1_420_000_000)])
    assert probe._parse_metadata(data)["parameter_count"] == 1_420_000_000


def test_parse_metadata_stops_early_once_all_three_are_found():
    """Real files put large tokenizer arrays after general.architecture/name/parameter_count. This confirms
    the early-break path doesn't choke on what follows once all three are found: a malformed trailing KV would raise if
    reached.
    """
    data = _build_gguf(
        [
            _kv_string("general.architecture", "llama"),
            _kv_string("general.name", "some-model"),
            _kv_u64("general.parameter_count", 7_000_000_000),
            _kv_string("trailing.unused", "value"),
        ]
    )
    assert probe._parse_metadata(data) == {
        "architecture": "llama",
        "name": "some-model",
        "parameter_count": 7_000_000_000,
    }


def test_parse_metadata_skips_bool_and_array_values_to_reach_a_later_key():
    data = _build_gguf(
        [
            _kv_bool("clip.has_text_encoder", False),
            _kv_u32_array("clip.vision.image_mean", [0, 1, 2]),
            _kv_string_array("tokenizer.ggml.tokens", ["a", "bb", "ccc"]),
            _kv_string("general.architecture", "clip"),
        ]
    )
    assert probe._parse_metadata(data)["architecture"] == "clip"


def test_parse_metadata_leaves_name_none_when_only_architecture_is_present():
    data = _build_gguf([_kv_string("general.architecture", "mistral3"), _kv_bool("some.other.flag", True)])
    assert probe._parse_metadata(data) == {
        "architecture": "mistral3",
        "name": None,
        "parameter_count": None,
    }


def test_parse_metadata_raises_buffer_exhausted_when_truncated_mid_value():
    full = _build_gguf([_kv_string("general.architecture", "phi2")])
    with pytest.raises(probe._BufferExhausted):
        probe._parse_metadata(full[:-3])


# probe_metadata (network, faked)


class _FakeResponse:
    def __init__(self, status_code: int, content: bytes = b""):
        self.status_code = status_code
        self.content = content


class _FakeAsyncClient:
    def __init__(self, responses, calls, raise_exc=None):
        self._responses = list(responses)
        self._calls = calls
        self._raise_exc = raise_exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def get(self, url: str, headers: dict):
        self._calls.append((url, headers))
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._responses.pop(0)


def _patch_client(monkeypatch, calls, responses=(), raise_exc=None):
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeAsyncClient(responses, calls, raise_exc=raise_exc))


@pytest.mark.asyncio
async def test_probe_metadata_returns_the_real_architecture_and_name(monkeypatch):
    calls: list = []
    data = _build_gguf([_kv_string("general.architecture", "phi2"), _kv_string("general.name", "moondream2")])
    _patch_client(monkeypatch, calls, responses=[_FakeResponse(206, data)])

    result = await probe.probe_metadata("moondream/moondream-2b-2025-04-14-4bit", "text.gguf", proxy_url=None)

    assert result == {"architecture": "phi2", "name": "moondream2", "parameter_count": None}
    url, headers = calls[0]
    assert url == "https://huggingface.co/moondream/moondream-2b-2025-04-14-4bit/resolve/main/text.gguf"
    assert headers["Range"] == f"bytes=0-{probe._FIRST_RANGE_BYTES - 1}"


@pytest.mark.asyncio
async def test_probe_metadata_retries_with_a_larger_range_once_the_first_is_truncated(monkeypatch):
    """A first-pass buffer running out mid-parse (_BufferExhausted) must trigger exactly one retry at the larger
    fallback range, not a silent give-up: unusual repos can have more metadata before general.architecture (2MB covers
    the common case; see gguf_probe).
    """
    calls: list = []
    full = _build_gguf([_kv_string("general.architecture", "qwen2")])
    truncated = full[:-2]
    _patch_client(monkeypatch, calls, responses=[_FakeResponse(206, truncated), _FakeResponse(206, full)])

    result = await probe.probe_metadata("org/repo", "model.gguf", proxy_url=None)

    assert result == {"architecture": "qwen2", "name": None, "parameter_count": None}
    assert len(calls) == 2
    assert calls[1][1]["Range"] == f"bytes=0-{probe._FALLBACK_RANGE_BYTES - 1}"


@pytest.mark.asyncio
async def test_probe_metadata_returns_none_when_still_truncated_at_the_fallback_range(monkeypatch):
    calls: list = []
    truncated = _build_gguf([_kv_string("general.architecture", "qwen2")])[:-2]
    _patch_client(monkeypatch, calls, responses=[_FakeResponse(206, truncated), _FakeResponse(206, truncated)])

    result = await probe.probe_metadata("org/repo", "model.gguf", proxy_url=None)

    assert result is None
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_probe_metadata_returns_none_for_a_non_gguf_file(monkeypatch):
    calls: list = []
    _patch_client(monkeypatch, calls, responses=[_FakeResponse(206, b"not a gguf file at all" + b"\x00" * 20)])

    assert await probe.probe_metadata("org/repo", "readme.txt", proxy_url=None) is None
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_probe_metadata_returns_none_on_a_404(monkeypatch):
    calls: list = []
    _patch_client(monkeypatch, calls, responses=[_FakeResponse(404)])

    assert await probe.probe_metadata("org/repo", "missing.gguf", proxy_url=None) is None


@pytest.mark.asyncio
async def test_probe_metadata_returns_none_on_a_network_error(monkeypatch):
    calls: list = []
    _patch_client(monkeypatch, calls, raise_exc=httpx.ConnectTimeout("timed out"))

    assert await probe.probe_metadata("org/repo", "model.gguf", proxy_url=None) is None


@pytest.mark.asyncio
async def test_probe_metadata_returns_none_instead_of_raising_on_malformed_binary_content(monkeypatch):
    """Untrusted internet binary: an unrecognized value_type (not a real GGUF, maybe corrupt or adversarial) must give
    "couldn't determine it", never crash the "Add" action (see probe_metadata).
    """
    calls: list = []
    corrupt = _build_gguf([_gguf_string("general.architecture") + struct.pack("<I", 255)])
    _patch_client(monkeypatch, calls, responses=[_FakeResponse(206, corrupt)])

    assert await probe.probe_metadata("org/repo", "model.gguf", proxy_url=None) is None
