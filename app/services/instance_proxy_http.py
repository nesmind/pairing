"""Plumbing helpers for instance_proxy.py: exemption/job-routing path matching, header filtering and ASGI
body reading. Split out for the file-size rule, like instance_process.py from instance_service.py.
"""

import re

import httpx

# Endpoints whose state exists only on the primary and must never be proxied, in either mode:
#   - the local-instances supervisor's tracking/reconcile state
#   - db_admin's switch_database, which changes only *that process's* connection
#   - the proxy-mode setting
#   - ollama_process's restart (one shared Ollama; it must fire once, not twice on a random sibling)
#   - /health, which must answer for the port asked, or health checks mean nothing
# /static/ is exempt as an optimization (identical files everywhere, no need for a hop).
EXEMPT_PREFIXES = (
    "/api/settings/instances",
    "/api/settings/database",
    "/api/settings/proxy-mode",
    # Ollama process state (ollama_process.py) exists only on the primary, as for ComfyUI below. Covers
    # local-mode start/stop/status and the old "Ollama concurrency" config, now under the same prefix.
    "/api/settings/ollama",
    # ComfyUI process state (comfyui_service.py) exists only on the primary, like the other supervision entries.
    # Without this, a start/stop/status load-balanced to a sibling would 400 (comfyui_service's IS_PRIMARY guard)
    # or read/write that *sibling's* own comfyui.json.
    "/api/settings/comfyui",
    # stable-diffusion.cpp's sd-server: primary-only process state, as ComfyUI.
    "/api/settings/sdcpp",
    # Matricxon process state (matricxon_process.py) exists only on the primary, like Ollama/ComfyUI. It was
    # missing: a start/stop landing on a sibling 400'd with "Matricxon can only be managed from the primary
    # instance", so the button worked about half the time.
    "/api/settings/matricxon",
    "/health",
    "/static/",
)


def is_exempt(path: str) -> bool:
    return path.startswith(EXEMPT_PREFIXES)


# A RAG upload job (document_upload.py) lives in the in-memory dict of the instance that started it. Its id
# is prefixed with that instance's index so a poll routes straight back, with no proxy-side state or body inspection.
_UPLOAD_JOB_RE = re.compile(r"^/api/documents/upload/(\d+)-[0-9a-f]{32}$")


def job_target_index(path: str) -> int | None:
    match = _UPLOAD_JOB_RE.match(path)
    return int(match.group(1)) if match else None


# Small JSON bodies (most requests: chat, login, settings) are buffered so a failure before the first byte can
# retry on another instance. Larger ones (e.g. an upload) stream unbuffered with a single attempt: retrying
# would need buffering many MB or re-reading a drained ASGI receive stream.
MAX_RETRY_BODY_BYTES = 262_144

PROXY_TIMEOUT = httpx.Timeout(300.0, connect=5.0)

# Drop "x-forwarded-for" from the client's headers, not just "host". Otherwise a forged X-Forwarded-For
# precedes the trusted one appended below, and Starlette's Headers.get (used by
# auth_service.resolve_client_ip on the sibling) returns the first match: the forged value, defeating the
# loopback-trust check and the per-IP login rate limiter.
_DROP_REQUEST_HEADERS = {"host", "x-forwarded-for"}
_DROP_RESPONSE_HEADERS = {"content-length", "transfer-encoding", "connection"}


def client_ip(scope) -> str:
    client = scope.get("client")
    return client[0] if client else "unknown"


def content_length(scope) -> int | None:
    for key, value in scope["headers"]:
        if key.decode("latin-1").lower() == "content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


def filter_request_headers(raw_headers: list[tuple[bytes, bytes]], ip: str) -> list[tuple[str, str]]:
    headers = [
        (k.decode("latin-1"), v.decode("latin-1"))
        for k, v in raw_headers
        if k.decode("latin-1").lower() not in _DROP_REQUEST_HEADERS
    ]
    headers.append(("X-Forwarded-For", ip))
    return headers


def filter_response_headers(raw_headers) -> list[tuple[bytes, bytes]]:
    return [
        (k.encode("latin-1"), v.encode("latin-1")) for k, v in raw_headers if k.lower() not in _DROP_RESPONSE_HEADERS
    ]


def target_url(scope, port: int) -> str:
    query = scope.get("query_string", b"")
    url = f"http://localhost:{port}{scope['path']}"
    return f"{url}?{query.decode('latin-1')}" if query else url


async def read_full_body(receive) -> bytes:
    chunks = []
    more_body = True
    while more_body:
        message = await receive()
        if message["type"] != "http.request":
            break
        chunks.append(message.get("body", b""))
        more_body = message.get("more_body", False)
    return b"".join(chunks)


async def stream_body(receive):
    more_body = True
    while more_body:
        message = await receive()
        if message["type"] != "http.request":
            break
        yield message.get("body", b"")
        more_body = message.get("more_body", False)
