"""Unit tests for the cancel/progress parts of app/routers/image_generation.py — called directly."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.routers import image_generation
from app.schemas import ImageGenerationRequest
from app.services import image_generation_service, sdcpp_progress


async def _job(db, user, monkeypatch, status="queued"):
    monkeypatch.setattr(image_generation_service, "_schedule", lambda _job_id: None)
    job = await image_generation_service.create_job(db, user.id, ImageGenerationRequest(prompt="a cat", checkpoint="m"))
    job.status = status
    await db.commit()
    return job


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["queued", "running"])
async def test_cancel_marks_an_unfinished_job_cancelled(db, user, monkeypatch, status):
    job = await _job(db, user, monkeypatch, status)
    result = await image_generation.cancel_job(job.id, db=db, user=user)
    assert result.status == "cancelled"


@pytest.mark.asyncio
async def test_cancel_is_a_noop_for_a_finished_job(db, user, monkeypatch):
    job = await _job(db, user, monkeypatch, "complete")
    assert (await image_generation.cancel_job(job.id, db=db, user=user)).status == "complete"


@pytest.mark.asyncio
async def test_cancel_hides_other_users_jobs_behind_a_404(db, user, monkeypatch):
    job = await _job(db, user, monkeypatch)
    with pytest.raises(HTTPException) as exc_info:
        await image_generation.cancel_job(job.id, db=db, user=SimpleNamespace(id="someone-else"))
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_get_job_reports_live_progress_for_a_running_sdcpp_job(db, user, monkeypatch, tmp_path):
    job = await _job(db, user, monkeypatch, "running")
    job.log_offset = 0
    await db.commit()
    log = tmp_path / "sdcpp.log"
    log.write_text("  |====>   | 2/4 - 10.0s/it\r")
    real = sdcpp_progress.read_progress
    monkeypatch.setattr(sdcpp_progress, "read_progress", lambda offset: real(offset, log))

    out = await image_generation.get_job(job.id, db=db, user=user)

    assert out.stage == "Sampling · step 2 of 4" and out.eta_seconds == 20 and out.progress > 25


@pytest.mark.asyncio
async def test_get_job_has_no_progress_when_not_running(db, user, monkeypatch):
    job = await _job(db, user, monkeypatch, "queued")
    out = await image_generation.get_job(job.id, db=db, user=user)
    assert out.progress is None and out.stage is None
