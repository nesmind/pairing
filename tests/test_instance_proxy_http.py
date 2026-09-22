"""Unit tests for the pure helpers in
app/services/instance_proxy_http.py — the ASGI middleware itself
(app/services/instance_proxy.py) needs a real running instance to
exercise (see this project's live end-to-end verification instead,
matching how the note-slot/scroll/multi-instance features earlier were
verified: real processes, real curl, not a new TestClient/ASGI-
integration test pattern this repo doesn't otherwise use)."""

from app.services.instance_proxy_http import (
    filter_request_headers,
    filter_response_headers,
    is_exempt,
    job_target_index,
)


def test_is_exempt_matches_every_pinned_to_primary_prefix():
    for path in (
        "/api/settings/instances",
        "/api/settings/database",
        "/api/settings/database/test-connection",
        "/api/settings/proxy-mode",
        "/api/settings/ollama",
        "/api/settings/ollama/start",
        "/api/settings/comfyui",
        "/api/settings/comfyui/start",
        "/api/settings/matricxon",
        "/api/settings/matricxon/start",
        "/health",
        "/static/js/app.js",
    ):
        assert is_exempt(path), path


def test_is_exempt_false_for_ordinary_routes():
    for path in ("/api/chat/abc123/stream", "/", "/api/documents/upload", "/api/conversations"):
        assert not is_exempt(path), path


def test_job_target_index_parses_a_prefixed_job_id():
    assert job_target_index("/api/documents/upload/2-" + "a" * 32) == 2
    assert job_target_index("/api/documents/upload/0-" + "f" * 32) == 0


def test_job_target_index_none_for_non_matching_paths():
    assert job_target_index("/api/documents/upload") is None
    assert job_target_index("/api/documents/upload/" + "a" * 32) is None  # no prefix, pre-feature job id shape
    assert job_target_index("/api/conversations/abc") is None


def test_filter_request_headers_drops_host_and_adds_forwarded_for():
    raw = [(b"Host", b"localhost:8000"), (b"Cookie", b"session=abc"), (b"Accept", b"*/*")]
    result = dict(filter_request_headers(raw, "203.0.113.5"))
    assert "Host" not in result and "host" not in result
    assert result["Cookie"] == "session=abc"
    assert result["Accept"] == "*/*"
    assert result["X-Forwarded-For"] == "203.0.113.5"


def test_filter_request_headers_drops_a_client_supplied_forwarded_for():
    """Security regression test: a client's own X-Forwarded-For must
    never reach a sibling instance at all — not just have the trusted
    one appended after it. Starlette's Headers on the receiving side
    could otherwise return the *attacker's* forged value first (or a
    dict-building implementation could let it silently win outright),
    letting a client rotate this header to bypass
    auth_service.resolve_client_ip's loopback-trust check and defeat the
    per-IP login rate limiter entirely."""
    raw = [(b"X-Forwarded-For", b"203.0.113.9"), (b"Cookie", b"session=abc")]
    result = filter_request_headers(raw, "127.0.0.1")
    values = [v for k, v in result if k.lower() == "x-forwarded-for"]
    assert values == ["127.0.0.1"]  # only the trusted one, never the client's forged value


def test_filter_response_headers_drops_hop_by_hop_headers():
    raw = [("Content-Length", "42"), ("Transfer-Encoding", "chunked"), ("Connection", "keep-alive"), ("X-Custom", "y")]
    result = filter_response_headers(raw)
    names = {k.decode("latin-1").lower() for k, _ in result}
    assert names == {"x-custom"}
