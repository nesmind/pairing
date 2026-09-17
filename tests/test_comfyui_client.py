"""Unit tests for app/services/comfyui_client.py — the ComfyUI HTTP
wrapper. Every httpx call is monkeypatched; no real ComfyUI instance is
reachable in this environment. submit_job/list_checkpoints route through
app.services.comfyui_pool's HostPool (see tests/test_comfyui_pool.py and
tests/test_server_pool.py for that logic itself) — with only the default
single configured host in play here, pool routing is transparent to
these tests beyond submit_job now returning (host, prompt_id)."""

import copy

import httpx
import pytest

from app.services import comfyui_client
from app.services.comfyui_client import ComfyUIError


def test_build_workflow_substitutes_every_param():
    params = {
        "checkpoint": "sd15.safetensors",
        "prompt": "a cat",
        "negative_prompt": "blurry",
        "width": 768,
        "height": 640,
        "steps": 30,
        "cfg": 8.5,
        "seed": 42,
    }
    workflow = comfyui_client._build_workflow(params)

    assert workflow["4"]["inputs"]["ckpt_name"] == "sd15.safetensors"
    assert workflow["5"]["inputs"]["width"] == 768
    assert workflow["5"]["inputs"]["height"] == 640
    assert workflow["6"]["inputs"]["text"] == "a cat"
    assert workflow["7"]["inputs"]["text"] == "blurry"
    assert workflow["3"]["inputs"]["steps"] == 30
    assert workflow["3"]["inputs"]["cfg"] == 8.5
    assert workflow["3"]["inputs"]["seed"] == 42


def test_build_workflow_defaults_a_missing_negative_prompt_to_empty_string():
    params = {
        "checkpoint": "x",
        "prompt": "a cat",
        "negative_prompt": None,
        "width": 512,
        "height": 512,
        "steps": 20,
        "cfg": 7,
        "seed": 0,
    }
    workflow = comfyui_client._build_workflow(params)
    assert workflow["7"]["inputs"]["text"] == ""


def test_build_workflow_never_mutates_the_shared_template():
    before = copy.deepcopy(comfyui_client._WORKFLOW_TEMPLATE)
    comfyui_client._build_workflow(
        {
            "checkpoint": "x",
            "prompt": "mutate me",
            "negative_prompt": "bad",
            "width": 999,
            "height": 999,
            "steps": 1,
            "cfg": 1,
            "seed": 1,
        }
    )
    assert comfyui_client._WORKFLOW_TEMPLATE == before


class _FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code
        self.content = b"fake-bytes"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=httpx.Request("GET", "http://x"), response=self)

    def json(self):
        return self._json


class _FakeAsyncClient:
    def __init__(self, response=None, raise_error=None):
        self._response = response
        self._raise_error = raise_error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def get(self, *_args, **_kwargs):
        if self._raise_error:
            raise self._raise_error
        return self._response

    async def post(self, *_args, **_kwargs):
        if self._raise_error:
            raise self._raise_error
        return self._response


@pytest.mark.asyncio
async def test_submit_job_returns_the_host_and_prompt_id(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeAsyncClient(_FakeResponse({"prompt_id": "abc123"})))
    params = {
        "checkpoint": "x",
        "prompt": "p",
        "negative_prompt": None,
        "width": 512,
        "height": 512,
        "steps": 1,
        "cfg": 1,
        "seed": 0,
    }
    host, prompt_id = await comfyui_client.submit_job(params)
    assert prompt_id == "abc123"
    assert host  # whichever host the pool picked


@pytest.mark.asyncio
async def test_submit_job_wraps_http_errors(monkeypatch):
    error = httpx.ConnectError("refused", request=httpx.Request("POST", "http://x"))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeAsyncClient(raise_error=error))
    with pytest.raises(ComfyUIError):
        await comfyui_client.submit_job(
            {
                "checkpoint": "x",
                "prompt": "p",
                "negative_prompt": None,
                "width": 512,
                "height": 512,
                "steps": 1,
                "cfg": 1,
                "seed": 0,
            }
        )


@pytest.mark.asyncio
async def test_get_history_returns_none_while_not_yet_finished(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeAsyncClient(_FakeResponse({})))
    assert await comfyui_client.get_history("http://h1:8188", "abc123") is None


@pytest.mark.asyncio
async def test_get_history_returns_the_entry_once_present(monkeypatch):
    entry = {"outputs": {"9": {"images": [{"filename": "f.png"}]}}}
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeAsyncClient(_FakeResponse({"abc123": entry})))
    assert await comfyui_client.get_history("http://h1:8188", "abc123") == entry


@pytest.mark.asyncio
async def test_get_history_wraps_http_errors(monkeypatch):
    error = httpx.ConnectError("refused", request=httpx.Request("GET", "http://x"))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeAsyncClient(raise_error=error))
    with pytest.raises(ComfyUIError):
        await comfyui_client.get_history("http://h1:8188", "abc123")


@pytest.mark.asyncio
async def test_fetch_image_bytes_returns_the_raw_content(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeAsyncClient(_FakeResponse({})))
    assert await comfyui_client.fetch_image_bytes("http://h1:8188", "f.png", "", "output") == b"fake-bytes"


@pytest.mark.asyncio
async def test_fetch_image_bytes_wraps_http_errors(monkeypatch):
    error = httpx.ConnectError("refused", request=httpx.Request("GET", "http://x"))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeAsyncClient(raise_error=error))
    with pytest.raises(ComfyUIError):
        await comfyui_client.fetch_image_bytes("http://h1:8188", "f.png", "", "output")


@pytest.mark.asyncio
async def test_list_checkpoints_reads_the_enum_list(monkeypatch):
    shape = {"CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["a.safetensors", "b.safetensors"]]}}}}
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeAsyncClient(_FakeResponse(shape)))
    assert await comfyui_client.list_checkpoints() == ["a.safetensors", "b.safetensors"]


@pytest.mark.asyncio
async def test_list_checkpoints_raises_comfyui_error_on_unexpected_shape(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeAsyncClient(_FakeResponse({"unexpected": True})))
    with pytest.raises(ComfyUIError):
        await comfyui_client.list_checkpoints()
