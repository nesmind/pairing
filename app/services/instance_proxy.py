""" "Proxy mode" (Settings > System): when "local" (default; settings_service.get_proxy_mode / instance_pool's
cache), this ASGI middleware forwards a request to the instance instance_pool.pick_instance chooses
(possibly this primary) and streams the response back. With "proxy" or one instance it does nothing; an
external reverse proxy is expected to balance across each instance's port.

Exemption, job-routing and header helpers: instance_proxy_http.py (split for the file-size rule).

A raw ASGI middleware, not BaseHTTPMiddleware: the chat SSE stream (routers/chat.py) can run for minutes
and needs ASGI cancellation to propagate on client disconnect (chat_service.build_reply_stream), which
BaseHTTPMiddleware has a history of not doing for long streamed responses.

Register (app.add_middleware in app/main.py) *after* SessionMiddleware: Starlette makes the last-registered
middleware outermost, so a proxied request never touches this process's session/router; the target decodes
the shared-secret, stateless session cookie itself (auth_service: no server-side store, so no sticky routing).
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
        """Streams the proxied response through, never buffering it (long chat SSE or large uploads).

        Races forwarding the body against `receive` for http.disconnect and cancels forwarding on arrival. This
        is load-bearing: ASGI signals a disconnect via `receive`, never by making `send` raise, so a loop that
        only sends never learns the client left (an earlier version did this). Without it, abandoning a chat stream
        left the *upstream* generation running for minutes, holding the engine's single-request slot for a reply no
        one read, which build_reply_stream's cancellation contract is meant to prevent. Cancelling closes httpx's
        stream (`async with client.stream(...)` exits), dropping the upstream connection, which cancels its generation
        as a direct disconnect does.
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
                    forward_task.result()
