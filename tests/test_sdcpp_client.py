"""Unit tests for app/services/sdcpp_client.py — httpx is monkeypatched; no real sd-server is reachable here."""

import base64

import httpx
import pytest

from app.services import sdcpp_client
from app.services.sdcpp_client import SdCppError

_PARAMS = {
    "checkpoint": "m",
    "prompt": "a cat",
    "negative_prompt": None,
    "width": 256,
    "height": 320,
    "steps": 4,
    "cfg": 1.5,
    "seed": 7,
}


class _FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=httpx.Request("POST", "http://x"), response=self)

    def json(self):
        return self._json


class _FakeAsyncClient:
    def __init__(self, response=None, raise_error=None, sink=None):
        self._response, self._raise_error, self._sink = response, raise_error, sink

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def _reply(self, url, json=None):
        if self._sink is not None:
            self._sink.append((url, json))
        if self._raise_error:
            raise self._raise_error
        return self._response

    async def post(self, url, json=None, **_kw):
        return await self._reply(url, json)

    async def get(self, url, **_kw):
        return await self._reply(url)


@pytest.mark.asyncio
async def test_submit_job_posts_the_native_payload_and_returns_host_and_id(monkeypatch):
    sent = []
    response = _FakeResponse({"id": "job_1", "status": "queued"})
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeAsyncClient(response, sink=sent))

    host, job_id = await sdcpp_client.submit_job(_PARAMS)

    assert job_id == "job_1" and host
    url, payload = sent[0]
    assert url.endswith("/sdcpp/v1/img_gen")
    assert payload["prompt"] == "a cat" and payload["negative_prompt"] == ""
    assert (payload["width"], payload["height"], payload["seed"]) == (256, 320, 7)
    assert payload["sample_params"] == {"sample_steps": 4, "guidance": {"txt_cfg": 1.5}}


@pytest.mark.asyncio
async def test_submit_job_wraps_http_errors(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeAsyncClient(raise_error=httpx.ConnectError("down")))
    with pytest.raises(SdCppError):
        await sdcpp_client.submit_job(_PARAMS)


@pytest.mark.asyncio
async def test_get_job_returns_the_state_and_maps_a_gone_job_to_an_error(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeAsyncClient(_FakeResponse({"status": "generating"})))
    assert (await sdcpp_client.get_job("http://h", "j"))["status"] == "generating"
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeAsyncClient(_FakeResponse({}, status_code=410)))
    with pytest.raises(SdCppError, match="no longer knows"):
        await sdcpp_client.get_job("http://h", "j")


def test_image_from_job_decodes_the_png_and_rejects_a_bad_shape():
    png = b"\x89PNG-bytes"
    job = {"result": {"images": [{"index": 0, "b64_json": base64.b64encode(png).decode()}]}}
    assert sdcpp_client.image_from_job(job) == png
    with pytest.raises(SdCppError):
        sdcpp_client.image_from_job({"result": None})


@pytest.mark.asyncio
async def test_cancel_job_posts_to_the_job_and_swallows_errors(monkeypatch):
    sent = []
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeAsyncClient(_FakeResponse({}), sink=sent))
    await sdcpp_client.cancel_job("http://h:1", "job_9")
    assert sent[0][0] == "http://h:1/sdcpp/v1/jobs/job_9/cancel"
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeAsyncClient(raise_error=httpx.ConnectError("down")))
    await sdcpp_client.cancel_job("http://h:1", "job_9")  # must not raise


@pytest.mark.asyncio
async def test_list_checkpoints_returns_model_names(monkeypatch):
    response = _FakeResponse([{"model_name": "sd-turbo", "title": "sd-turbo"}])
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeAsyncClient(response))
    assert await sdcpp_client.list_checkpoints() == ["sd-turbo"]


@pytest.mark.asyncio
async def test_list_checkpoints_wraps_http_errors(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeAsyncClient(raise_error=httpx.ConnectError("down")))
    with pytest.raises(SdCppError):
        await sdcpp_client.list_checkpoints()
