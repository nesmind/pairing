"""Image-to-image: request validation, the in-memory source image hand-off, and what is sent to sd-server."""

import base64
import struct
import zlib

import httpx
import pytest
from pydantic import ValidationError

from app.models import ImageGenerationJob
from app.schemas import ImageGenerationRequest
from app.services import image_engine_service, image_generation_service, image_init, sdcpp_client


def _png(width: int, height: int) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    raw = b"".join(b"\x00" + b"\x80\x40\x20" * width for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def _b64(width: int = 128, height: int = 192) -> str:
    return base64.b64encode(_png(width, height)).decode()


def _request(**overrides) -> ImageGenerationRequest:
    base = {"prompt": "a cat", "checkpoint": "m.gguf", "width": 128, "height": 192}
    return ImageGenerationRequest(**{**base, **overrides})


# ---- request validation -------------------------------------------------------------------------------------------


def test_text_to_image_is_the_default_and_drops_any_stray_source_image():
    request = _request(init_image=_b64())
    assert request.mode == "text_to_image" and request.init_image is None


def test_image_to_image_accepts_a_png_of_exactly_the_requested_size():
    request = _request(mode="image_to_image", init_image=_b64(), strength=0.5)
    assert request.init_image and request.strength == 0.5


@pytest.mark.parametrize(
    ("init_image", "message"),
    [
        (None, "needs a source image"),
        ("!!!not base64!!!", "not valid base64"),
        (base64.b64encode(b"GIF89a-not-a-png-at-all-padding").decode(), "must be a PNG"),
        (_b64(64, 64), "64x64 but the request is 128x192"),
    ],
)
def test_image_to_image_rejects_a_missing_or_unusable_source(init_image, message):
    with pytest.raises(ValidationError, match=message):
        _request(mode="image_to_image", init_image=init_image)


@pytest.mark.parametrize("strength", [0, -0.1, 1.5])
def test_strength_must_be_in_0_to_1(strength):
    with pytest.raises(ValidationError):
        _request(mode="image_to_image", init_image=_b64(), strength=strength)


# ---- the job row and the in-memory hand-off -----------------------------------------------------------------------


@pytest.fixture(autouse=True)
def no_background_run(monkeypatch):
    monkeypatch.setattr(image_generation_service, "_schedule", lambda _job_id: None)
    monkeypatch.setattr(image_generation_service, "_POLL_INTERVAL_SECONDS", 0)
    image_init._pending.clear()


@pytest.mark.asyncio
async def test_create_job_records_mode_and_strength_and_keeps_the_picture_off_the_row(db, user):
    job = await image_generation_service.create_job(
        db, user.id, _request(mode="image_to_image", init_image=_b64(), strength=0.4)
    )
    assert (job.mode, job.strength) == ("image_to_image", 0.4)
    assert image_init._pending[job.id] == _png(128, 192)

    plain = await image_generation_service.create_job(db, user.id, _request())
    assert (plain.mode, plain.strength) == ("text_to_image", None) and plain.id not in image_init._pending


def test_init_params_consumes_the_picture_once():
    job = ImageGenerationJob(id="j1", mode="image_to_image", strength=0.3)
    image_init.stash("j1", _b64())
    params = image_init.init_params(job, "sdcpp")
    assert params == {"init_image": _b64(), "strength": 0.3}
    with pytest.raises(ValueError, match="no longer available"):
        image_init.init_params(job, "sdcpp")


def test_init_params_is_empty_for_text_to_image_and_refuses_other_engines():
    assert image_init.init_params(ImageGenerationJob(id="j2", mode="text_to_image"), "sdcpp") == {}
    image_init.stash("j3", _b64())
    with pytest.raises(ValueError, match="stable-diffusion.cpp"):
        image_init.init_params(ImageGenerationJob(id="j3", mode="image_to_image", strength=0.5), "comfyui")
    assert "j3" not in image_init._pending  # consumed even when refused


@pytest.mark.asyncio
async def test_run_job_sends_the_source_and_strength_to_sdcpp(db, user, tmp_path, monkeypatch):
    monkeypatch.setattr(image_generation_service, "IMAGES_DIR", tmp_path)
    await image_engine_service.set_active_image_engine(db, "sdcpp")
    seen = {}

    async def fake_submit(params):
        seen.update(params)
        return "http://h1:8189", "job_1"

    async def fake_get(_host, _job_id):
        return {"status": "completed", "result": {"images": [{"b64_json": base64.b64encode(b"\x89PNG-out").decode()}]}}

    monkeypatch.setattr(sdcpp_client, "submit_job", fake_submit)
    monkeypatch.setattr(sdcpp_client, "get_job", fake_get)
    job = await image_generation_service.create_job(
        db, user.id, _request(mode="image_to_image", init_image=_b64(), strength=0.45)
    )
    await image_generation_service._run_job(job.id)

    await db.refresh(job)
    assert job.status == "complete"
    assert seen["init_image"] == _b64() and seen["strength"] == 0.45 and job.id not in image_init._pending


@pytest.mark.asyncio
async def test_run_job_errors_when_the_source_image_is_gone(db, user, monkeypatch):
    await image_engine_service.set_active_image_engine(db, "sdcpp")
    job = await image_generation_service.create_job(
        db, user.id, _request(mode="image_to_image", init_image=_b64(), strength=0.5)
    )
    image_init._pending.clear()  # as after a server restart
    await image_generation_service._run_job(job.id)
    await db.refresh(job)
    assert job.status == "error" and "no longer available" in job.error_message


@pytest.mark.asyncio
async def test_a_job_cancelled_before_it_runs_releases_its_picture(db, user):
    job = await image_generation_service.create_job(
        db, user.id, _request(mode="image_to_image", init_image=_b64(), strength=0.5)
    )
    job.status = "cancelled"
    await db.commit()
    await image_generation_service._run_job(job.id)
    assert job.id not in image_init._pending


# ---- what sd-server receives --------------------------------------------------------------------------------------


class _Response:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {"id": "job_1"}


class _Client:
    def __init__(self, sink):
        self._sink = sink

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def post(self, url, json):
        self._sink.append(json)
        return _Response()


@pytest.mark.asyncio
@pytest.mark.parametrize("extra", [{}, {"init_image": "QUJD", "strength": 0.6}])
async def test_submit_job_adds_init_image_and_strength_only_for_image_to_image(monkeypatch, extra):
    sent = []
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _Client(sent))
    params = {"prompt": "p", "width": 64, "height": 64, "seed": 1, "steps": 4, "cfg": 1.0, **extra}
    await sdcpp_client.submit_job(params)
    payload = sent[0]
    assert ("init_image" in payload) == bool(extra)
    if extra:
        assert payload["init_image"] == "QUJD" and payload["strength"] == 0.6
