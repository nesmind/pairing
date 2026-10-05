"""Unit tests for app/services/image_generation_service.py — the
DB-row-as-source-of-truth generation lifecycle (queued -> running ->
complete/error). _run_job is awaited directly (bypassing _schedule's
asyncio.create_task) since it's fire-and-forget by design and this
module has no synchronizing broadcast queue the way
reply_generation_service does — the deterministic way to test its full
transition logic is to just await the same coroutine the background
task would otherwise run. Every comfyui_client call is monkeypatched;
no real ComfyUI instance is reachable in this environment."""

import os

import pytest

from app.schemas import ImageGenerationRequest, SdCppConfig
from app.services import comfyui_client, image_engine_service, image_generation_service, sdcpp_client
from app.services.comfyui_client import ComfyUIError
from app.services.sdcpp_client import SdCppError


@pytest.fixture(autouse=True)
async def _comfyui_engine(db, monkeypatch):
    """The default engine is stable-diffusion.cpp and ComfyUI is switched off in production (ahead of its removal);
    the ComfyUI-path tests below re-enable it and pin it."""
    monkeypatch.setattr(image_engine_service, "ENABLED_IMAGE_ENGINES", ("sdcpp", "comfyui"))
    await image_engine_service.set_active_image_engine(db, "comfyui")


def _request(**overrides) -> ImageGenerationRequest:
    return ImageGenerationRequest(
        prompt="a cat",
        checkpoint="sd15.safetensors",
        **overrides,
    )


@pytest.mark.asyncio
async def test_create_job_writes_a_queued_row(db, user, monkeypatch):
    # Prevent the real background task from racing this test's own
    # assertions — schedule is exercised separately below.
    monkeypatch.setattr(image_generation_service, "_schedule", lambda _job_id: None)

    job = await image_generation_service.create_job(db, user.id, _request())

    assert job.status == "queued"
    assert job.owner_id == user.id
    assert job.prompt == "a cat"
    assert job.checkpoint == "sd15.safetensors"


@pytest.mark.asyncio
async def test_create_job_picks_a_random_seed_when_seed_is_negative(db, user, monkeypatch):
    monkeypatch.setattr(image_generation_service, "_schedule", lambda _job_id: None)
    job = await image_generation_service.create_job(db, user.id, _request(seed=-1))
    assert job.seed >= 0


@pytest.mark.asyncio
async def test_create_job_keeps_an_explicit_non_negative_seed(db, user, monkeypatch):
    monkeypatch.setattr(image_generation_service, "_schedule", lambda _job_id: None)
    job = await image_generation_service.create_job(db, user.id, _request(seed=12345))
    assert job.seed == 12345


@pytest.mark.asyncio
async def test_run_job_drives_a_full_success_through_to_complete(db, user, tmp_path, monkeypatch):
    monkeypatch.setattr(image_generation_service, "IMAGES_DIR", tmp_path)
    monkeypatch.setattr(image_generation_service, "_schedule", lambda _job_id: None)
    monkeypatch.setattr(image_generation_service, "_POLL_INTERVAL_SECONDS", 0)

    async def fake_submit_job(_params):
        return "http://h1:8188", "prompt-1"

    async def fake_get_history(_host, _prompt_id):
        return {"outputs": {comfyui_client.SAVE_IMAGE_NODE_ID: {"images": [{"filename": "out.png", "type": "output"}]}}}

    async def fake_fetch_image_bytes(_host, _filename, _subfolder, _type):
        return b"\x89PNG-fake-bytes"

    monkeypatch.setattr(comfyui_client, "submit_job", fake_submit_job)
    monkeypatch.setattr(comfyui_client, "get_history", fake_get_history)
    monkeypatch.setattr(comfyui_client, "fetch_image_bytes", fake_fetch_image_bytes)

    job = await image_generation_service.create_job(db, user.id, _request())
    await image_generation_service._run_job(job.id)

    # A separate AsyncSessionLocal() did the writes (see _run_job's own
    # docstring) — this test's own `db` session has `job` cached in its
    # identity map from create_job's own insert, and expire_on_commit=False
    # means a plain re-SELECT wouldn't re-read it; db.refresh() forces a
    # real read of the row's current state instead.
    await db.refresh(job)
    assert job.status == "complete"
    assert job.comfy_prompt_id == "prompt-1"
    assert job.image_path == f"{user.id}/{job.id}.png"
    assert (tmp_path / user.id / f"{job.id}.png").read_bytes() == b"\x89PNG-fake-bytes"


@pytest.mark.asyncio
async def test_run_job_marks_error_when_submit_fails(db, user, monkeypatch):
    monkeypatch.setattr(image_generation_service, "_schedule", lambda _job_id: None)

    async def fake_submit_job(_params):
        raise ComfyUIError("ComfyUI is unreachable")

    monkeypatch.setattr(comfyui_client, "submit_job", fake_submit_job)

    job = await image_generation_service.create_job(db, user.id, _request())
    await image_generation_service._run_job(job.id)

    await db.refresh(job)
    assert job.status == "error"
    assert "unreachable" in job.error_message


@pytest.mark.asyncio
async def test_run_job_marks_timeout_when_polling_never_finishes(db, user, monkeypatch):
    monkeypatch.setattr(image_generation_service, "_schedule", lambda _job_id: None)
    monkeypatch.setattr(image_generation_service, "_POLL_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(image_generation_service, "_MAX_POLL_ATTEMPTS", 2)

    async def fake_submit_job(_params):
        return "http://h1:8188", "prompt-1"

    async def fake_get_history(_host, _prompt_id):
        return None  # never finishes

    monkeypatch.setattr(comfyui_client, "submit_job", fake_submit_job)
    monkeypatch.setattr(comfyui_client, "get_history", fake_get_history)

    job = await image_generation_service.create_job(db, user.id, _request())
    await image_generation_service._run_job(job.id)

    await db.refresh(job)
    assert job.status == "error"
    assert "timed out" in job.error_message.lower()


@pytest.mark.asyncio
async def test_run_job_returns_quietly_if_the_row_was_deleted():
    # No row exists for this id at all — must not raise.
    await image_generation_service._run_job("does-not-exist")


@pytest.mark.asyncio
async def test_mark_interrupted_jobs_as_errored_sweeps_queued_and_running(db, user, monkeypatch):
    monkeypatch.setattr(image_generation_service, "_schedule", lambda _job_id: None)
    queued_job = await image_generation_service.create_job(db, user.id, _request())
    running_job = await image_generation_service.create_job(db, user.id, _request())
    running_job.status = "running"
    complete_job = await image_generation_service.create_job(db, user.id, _request())
    complete_job.status = "complete"
    await db.commit()

    count = await image_generation_service.mark_interrupted_jobs_as_errored(db)

    assert count == 2
    await db.refresh(queued_job)
    await db.refresh(running_job)
    await db.refresh(complete_job)
    assert queued_job.status == "error"
    assert running_job.status == "error"
    assert complete_job.status == "complete"  # untouched


def _fake_sdcpp(monkeypatch, states, seen=None, cancelled=None):
    """submit_job returns job_1 on h1; get_job walks `states`; cancel_job records the call."""
    it = iter(states)

    async def fake_submit(params):
        if seen is not None:
            seen.update(params)
        return "http://h1:8189", "job_1"

    async def fake_get(_host, _job_id):
        return next(it)

    async def fake_cancel(host, job_id):
        if cancelled is not None:
            cancelled.append((host, job_id))

    monkeypatch.setattr(sdcpp_client, "submit_job", fake_submit)
    monkeypatch.setattr(sdcpp_client, "get_job", fake_get)
    monkeypatch.setattr(sdcpp_client, "cancel_job", fake_cancel)
    monkeypatch.setattr(image_generation_service, "_POLL_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(image_generation_service, "_schedule", lambda _job_id: None)


@pytest.mark.asyncio
async def test_run_job_uses_sdcpp_when_it_is_the_active_engine(db, user, tmp_path, monkeypatch):
    import base64

    monkeypatch.setattr(image_generation_service, "IMAGES_DIR", tmp_path)
    await image_engine_service.set_active_image_engine(db, "sdcpp")
    seen = {}
    done = {"status": "completed", "result": {"images": [{"b64_json": base64.b64encode(b"\x89PNG-sdcpp").decode()}]}}
    _fake_sdcpp(monkeypatch, [{"status": "queued"}, {"status": "generating"}, done], seen)

    job = await image_generation_service.create_job(db, user.id, _request(steps=4))
    await image_generation_service._run_job(job.id)

    await db.refresh(job)
    assert job.status == "complete"
    assert job.comfy_prompt_id == "job_1" and job.log_offset is not None
    assert seen["steps"] == 4 and seen["prompt"] == "a cat"
    assert (tmp_path / user.id / f"{job.id}.png").read_bytes() == b"\x89PNG-sdcpp"


@pytest.mark.asyncio
async def test_run_job_saves_into_the_configured_images_folder_per_user(db, user, tmp_path, monkeypatch):
    import base64

    default, custom = tmp_path / "default", tmp_path / "custom"
    default.mkdir(), custom.mkdir()
    monkeypatch.setattr(image_generation_service, "IMAGES_DIR", default)
    await image_engine_service.set_active_image_engine(db, "sdcpp")
    await image_engine_service.set_sdcpp_config(db, SdCppConfig(images_path=str(custom)))
    done = {"status": "completed", "result": {"images": [{"b64_json": base64.b64encode(b"\x89PNG-x").decode()}]}}
    _fake_sdcpp(monkeypatch, [done], {})

    job = await image_generation_service.create_job(db, user.id, _request())
    await image_generation_service._run_job(job.id)

    await db.refresh(job)
    assert job.status == "complete" and job.image_path == f"{user.id}/{job.id}.png"
    assert (custom / user.id / f"{job.id}.png").read_bytes() == b"\x89PNG-x"
    assert not any(default.iterdir())


@pytest.mark.asyncio
async def test_run_job_marks_error_when_the_sdcpp_job_fails(db, user, monkeypatch):
    await image_engine_service.set_active_image_engine(db, "sdcpp")
    _fake_sdcpp(monkeypatch, [{"status": "failed", "error": {"message": "out of memory"}}])
    job = await image_generation_service.create_job(db, user.id, _request())
    await image_generation_service._run_job(job.id)
    await db.refresh(job)
    assert job.status == "error" and "out of memory" in job.error_message


@pytest.mark.asyncio
async def test_run_job_marks_error_when_sdcpp_is_unreachable(db, user, monkeypatch):
    monkeypatch.setattr(image_generation_service, "_schedule", lambda _job_id: None)
    await image_engine_service.set_active_image_engine(db, "sdcpp")

    async def fake_submit(_params):
        raise SdCppError("sd-server is unreachable")

    monkeypatch.setattr(sdcpp_client, "submit_job", fake_submit)
    job = await image_generation_service.create_job(db, user.id, _request())
    await image_generation_service._run_job(job.id)
    await db.refresh(job)
    assert job.status == "error" and "unreachable" in job.error_message


@pytest.mark.asyncio
async def test_a_cancel_during_an_sdcpp_run_stops_the_engine_and_keeps_the_cancelled_status(
    db, user, tmp_path, monkeypatch
):
    monkeypatch.setattr(image_generation_service, "IMAGES_DIR", tmp_path)
    await image_engine_service.set_active_image_engine(db, "sdcpp")
    cancelled = []

    class _CancelOnSecondPoll:
        calls = 0

        def __iter__(self):
            return self

        def __next__(self):
            type(self).calls += 1
            return {"status": "generating"}

    _fake_sdcpp(monkeypatch, _CancelOnSecondPoll(), cancelled=cancelled)
    job = await image_generation_service.create_job(db, user.id, _request())
    original = image_generation_service._is_cancelled
    checks = {"n": 0}

    async def cancelling_check(session, row):
        checks["n"] += 1
        if checks["n"] == 3:  # the user clicks Cancel while it's generating
            await image_generation_service.cancel_job(session, row)
        return await original(session, row)

    monkeypatch.setattr(image_generation_service, "_is_cancelled", cancelling_check)
    await image_generation_service._run_job(job.id)

    await db.refresh(job)
    assert job.status == "cancelled" and job.image_path is None
    assert cancelled == [("http://h1:8189", "job_1")]


@pytest.mark.asyncio
async def test_a_job_cancelled_while_queued_never_starts(db, user, monkeypatch):
    monkeypatch.setattr(image_generation_service, "_schedule", lambda _job_id: None)

    def fail(*_a, **_kw):
        raise AssertionError("a cancelled job must not reach the engine")

    monkeypatch.setattr(sdcpp_client, "submit_job", fail)
    job = await image_generation_service.create_job(db, user.id, _request())
    await image_generation_service.cancel_job(db, job)
    await image_generation_service._run_job(job.id)
    await db.refresh(job)
    assert job.status == "cancelled"


@pytest.mark.asyncio
async def test_cancel_leaves_a_finished_job_unchanged(db, user, monkeypatch):
    monkeypatch.setattr(image_generation_service, "_schedule", lambda _job_id: None)
    job = await image_generation_service.create_job(db, user.id, _request())
    job.status = "complete"
    await db.commit()
    assert (await image_generation_service.cancel_job(db, job)).status == "complete"


@pytest.mark.asyncio
async def test_a_cancel_during_a_comfyui_run_interrupts_comfyui(db, user, monkeypatch):
    monkeypatch.setattr(image_generation_service, "_schedule", lambda _job_id: None)
    monkeypatch.setattr(image_generation_service, "_POLL_INTERVAL_SECONDS", 0)
    interrupted = []

    async def fake_submit_job(_params):
        return "http://h1:8188", "prompt-1"

    async def fake_get_history(_host, _prompt_id):
        return None  # never finishes

    async def fake_cancel(host, prompt_id):
        interrupted.append((host, prompt_id))

    monkeypatch.setattr(comfyui_client, "submit_job", fake_submit_job)
    monkeypatch.setattr(comfyui_client, "get_history", fake_get_history)
    monkeypatch.setattr(comfyui_client, "cancel_job", fake_cancel)
    job = await image_generation_service.create_job(db, user.id, _request())
    await image_generation_service.cancel_job(db, job)
    job.status = "queued"  # let it start, then cancel on its first poll
    await db.commit()
    original = image_generation_service._is_cancelled

    async def check(session, row):
        await image_generation_service.cancel_job(session, row)
        return await original(session, row)

    monkeypatch.setattr(image_generation_service, "_is_cancelled", check)
    await image_generation_service._run_job(job.id)
    await db.refresh(job)
    assert job.status == "cancelled" and interrupted == [("http://h1:8188", "prompt-1")]


@pytest.mark.parametrize("bad", ["relative/dir", "/definitely/not/here"])
def test_check_images_dir_rejects_relative_and_missing_folders(bad):
    with pytest.raises(ValueError, match="images folder"):
        image_engine_service.check_images_dir(bad)


def test_check_images_dir_accepts_blank_and_an_existing_writable_folder(tmp_path):
    image_engine_service.check_images_dir(None)
    image_engine_service.check_images_dir("")
    image_engine_service.check_images_dir(str(tmp_path))


def test_check_images_dir_rejects_a_read_only_folder(tmp_path):
    tmp_path.chmod(0o500)
    try:
        if os.access(tmp_path, os.W_OK):  # running as root: permissions don't apply
            pytest.skip("folder stays writable for this user")
        with pytest.raises(ValueError, match="not writable"):
            image_engine_service.check_images_dir(str(tmp_path))
    finally:
        tmp_path.chmod(0o700)


@pytest.mark.asyncio
async def test_images_root_and_find_image_follow_the_configured_folder(db, tmp_path):
    default, custom = tmp_path / "default", tmp_path / "custom"
    default.mkdir(), custom.mkdir()
    assert await image_engine_service.images_root(db, default) == default

    await image_engine_service.set_sdcpp_config(db, SdCppConfig(images_path=str(custom)))
    assert await image_engine_service.images_root(db, default) == custom

    (default / "u1").mkdir()
    (default / "u1" / "old.png").write_bytes(b"x")
    (custom / "u1").mkdir()
    (custom / "u1" / "new.png").write_bytes(b"y")
    assert await image_engine_service.find_image(db, default, "u1/new.png") == custom / "u1" / "new.png"
    assert await image_engine_service.find_image(db, default, "u1/old.png") == default / "u1" / "old.png"  # made before
    assert await image_engine_service.find_image(db, default, "u1/none.png") is None
