"""Unit tests for app/services/image_generation_service.py — the
DB-row-as-source-of-truth generation lifecycle (queued -> running ->
complete/error). _run_job is awaited directly (bypassing _schedule's
asyncio.create_task) since it's fire-and-forget by design and this
module has no synchronizing broadcast queue the way
reply_generation_service does — the deterministic way to test its full
transition logic is to just await the same coroutine the background
task would otherwise run. Every comfyui_client call is monkeypatched;
no real ComfyUI instance is reachable in this environment."""

import pytest

from app.schemas import ImageGenerationRequest
from app.services import comfyui_client, image_generation_service
from app.services.comfyui_client import ComfyUIError


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
