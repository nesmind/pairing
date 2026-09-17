"""
Pure/plumbing helpers for app/services/instance_proxy.py's ASGI
middleware — exemption/job-routing path matching, header filtering, and
ASGI request-body reading. Split out purely to stay under CLAUDE.md's
file-size rule, same reasoning app/services/instance_process.py was
split from instance_service.py for.
"""

import re

import httpx

# Endpoints whose state only exists meaningfully on the primary process
# itself — must never be proxied away, in either mode:
#   - the local-instances supervisor's own tracking/reconcile state
#   - db_admin's switch_database(), which only changes *that process's*
#     own live DB connection, not the whole fleet's
#   - this feature's own proxy-mode setting
#   - ollama_process's restart action — Ollama is one shared process for
#     every local instance; this must fire exactly once, not risk a
#     concurrent/duplicate restart if it landed on a random sibling
#   - /health, which must always answer for the specific port it was
#     asked about, or the whole health-check mechanism means nothing
# /static/ is exempted too, purely as an optimization — identical files
# on every instance, no reason to add a hop.
EXEMPT_PREFIXES = (
    "/api/settings/instances",
    "/api/settings/database",
    "/api/settings/proxy-mode",
    # Ollama process state (app/services/ollama_process.py) only exists
    # meaningfully on the primary — same reasoning as the ComfyUI entry
    # right below. Covers both local-mode start/stop/status and the old
    # "Ollama concurrency" config, now folded into the same prefix.
    "/api/settings/ollama",
    # ComfyUI process state (app/services/comfyui_service.py) only exists
    # meaningfully on the primary — same reasoning as the other
    # process-supervision entries above. Without this, a start/stop/status
    # call load-balanced onto a sibling would either 400 (comfyui_service's
    # own IS_PRIMARY guard correctly refusing to manage a process it
    # doesn't own) or silently read/write that *sibling's* own separate
    # comfyui.json instead of the primary's.
    "/api/settings/comfyui",
    "/health",
    "/static/",
)


def is_exempt(path: str) -> bool:
    return path.startswith(EXEMPT_PREFIXES)


# A RAG upload job (app/services/document_upload.py) lives only in the
# in-memory dict of whichever instance's POST /api/documents/upload
# started it — its id is prefixed with that instance's own index at
# creation time specifically so a later poll can be routed straight
# back to it deterministically, with no proxy-side state of its own and
# no need to inspect any response body.
_UPLOAD_JOB_RE = re.compile(r"^/api/documents/upload/(\d+)-[0-9a-f]{32}$")


def job_target_index(path: str) -> int | None:
    match = _UPLOAD_JOB_RE.match(path)
    return int(match.group(1)) if match else None


# Small JSON API bodies (the overwhelming majority of this app's
# requests — chat messages, login, settings saves) are worth buffering
# so a pre-first-byte connection failure can safely retry on a
# different instance. A body above this (e.g. a document upload) is
# streamed straight through unbuffered instead — retrying it would mean
# either buffering something potentially many MB (defeating true
# streaming) or re-reading an already-drained ASGI receive() stream
# (impossible), so those get a single attempt only.
MAX_RETRY_BODY_BYTES = 262_144

PROXY_TIMEOUT = httpx.Timeout(300.0, connect=5.0)

# "x-forwarded-for" must be dropped from the client's own headers, not
# just "host" — otherwise a client can send its own X-Forwarded-For and
# it rides along *ahead of* the trusted one appended below, and Starlette's
# Headers.get() (what app.services.auth_service.resolve_client_ip reads on
# the receiving sibling) returns the *first* match — the attacker's
# forged value, not the real one, silently defeating the loopback-trust
# check that value exists for (see resolve_client_ip's own docstring) and
# letting a client bypass the per-IP login rate limiter at will.
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
