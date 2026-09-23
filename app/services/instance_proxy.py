"""
"Proxy mode" (Settings > System): when set to "local" (the default —
see app.services.settings_service.get_proxy_mode/instance_pool's cached
copy of it), this ASGI middleware transparently forwards an incoming
request to whichever local instance app.services.instance_pool.
pick_instance() chooses (possibly this primary process itself), and
streams the response back. When set to "proxy", or when there's only
one instance, this middleware does nothing at all — an external
reverse proxy, if any, is expected to balance across each instance's
own port directly instead.

The pure exemption/job-routing/header-filtering helpers this relies on
live in app/services/instance_proxy_http.py (split out purely to stay
under CLAUDE.md's file-size rule).

A raw ASGI middleware, not Starlette's BaseHTTPMiddleware: the chat
SSE stream (app/routers/chat.py) can run for minutes and relies on
ASGI cancellation propagating cleanly when a client disconnects (see
app.services.chat_service.build_reply_stream's own docstring) —
BaseHTTPMiddleware has a known history of not propagating that
cleanly through long-lived streamed responses.

Must be registered (app.add_middleware in app/main.py) *after*
SessionMiddleware — Starlette's add_middleware() prepends to its
internal list and builds the stack by iterating it in reverse, so
whichever middleware is registered *last* ends up outermost (sees the
request first). Registering this one last means a request that gets
proxied away never touches this process's own session/router handling
at all — the target instance decodes the (shared-secret, stateless)
session cookie independently. See app/services/auth_service.py's own
docstring: sessions are a pure signed cookie, no server-side store, so
no sticky routing is needed for login state.
"""

import asyncio

import httpx

from app.config import IS_PRIMARY
from app.services import instance_pool, instance_process, instance_proxy_http as http_


class InstanceProxyMiddleware:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or not IS_PRIMARY or http_.is_exempt(scope["path"]):
            await self.app(scope, receive, send)
            return

        job_index = http_.job_target_index(scope["path"])
        if job_index is not None:
            if job_index == 0:
                await self.app(scope, receive, send)
            else:
                async with instance_pool.track_request(job_index):
                    await self._proxy_once(scope, receive, send, job_index)
            return

        if instance_pool.get_cached_proxy_mode() != "local":
            await self.app(scope, receive, send)
            return

        index = instance_pool.pick_instance()
        if index == 0:
            await self.app(scope, receive, send)
            return

        length = http_.content_length(scope)
        retryable = length is not None and length <= http_.MAX_RETRY_BODY_BYTES
        body = await http_.read_full_body(receive) if retryable else None

        started = False

        async def send_wrapper(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        async with instance_pool.track_request(index):
            try:
                await self._proxy_once(scope, receive, send_wrapper, index, body=body)
            except httpx.HTTPError:
                if started or not retryable:
                    raise
                instance_pool.mark_failure(index)
                fallback = instance_pool.pick_instance(exclude=frozenset({index}))
                if fallback == index:
                    raise
                async with instance_pool.track_request(fallback):
                    await self._proxy_once(scope, receive, send, fallback, body=body)
            else:
                instance_pool.mark_recovered(index)

    @staticmethod
    async def _proxy_once(scope, receive, send, index: int, body: bytes | None = None) -> None:
        """Streams the proxied response straight through — never reads
        it fully into memory first, so a long chat SSE reply or a large
        upload response isn't buffered or delayed.

        Races forwarding the response body against watching `receive()`
        for an http.disconnect message, and cancels forwarding the
        instant one arrives. This is load-bearing, not defensive
        padding: ASGI signals a client disconnect by sending that
        message through `receive()`, never by making `send()` raise —
        a loop that only calls `send()` while forwarding chunks (as an
        earlier version of this method did) never learns the client
        left at all. Confirmed live: without this, abandoning a chat
        stream after a few chunks left the *upstream* instance's own
        generation running to completion regardless — several minutes
        of the ML engine's single-request slot held hostage by a reply nobody
        was still reading, exactly the failure chat_service.
        build_reply_stream's own cancellation-propagation contract is
        supposed to prevent. Cancelling the forwarding task here closes
        httpx's stream (the `async with client.stream(...)` below exits
        on the way out), which drops the connection to the upstream
        instance — the same mechanism that cancels its own generation
        when a *direct* (non-proxied) client disconnects.
        """
        port = instance_process.port_for_index(index)
        url = http_.target_url(scope, port)
        headers = http_.filter_request_headers(scope["headers"], http_.client_ip(scope))
        content = body if body is not None else http_.stream_body(receive)

        async with httpx.AsyncClient(timeout=http_.PROXY_TIMEOUT) as client:
            async with client.stream(scope["method"], url, headers=headers, content=content) as resp:
                response_headers = http_.filter_response_headers(resp.headers.multi_items())
                await send({"type": "http.response.start", "status": resp.status_code, "headers": response_headers})

                async def forward_body():
                    async for chunk in resp.aiter_bytes():
                        await send({"type": "http.response.body", "body": chunk, "more_body": True})
                    await send({"type": "http.response.body", "body": b"", "more_body": False})

                async def watch_for_disconnect():
                    while True:
                        message = await receive()
                        if message["type"] == "http.disconnect":
                            return

                forward_task = asyncio.ensure_future(forward_body())
                disconnect_task = asyncio.ensure_future(watch_for_disconnect())
                done, pending = await asyncio.wait(
                    {forward_task, disconnect_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                if forward_task in done:
                    forward_task.result()  # re-raise any real forwarding error
