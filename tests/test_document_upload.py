"""Unit tests for app/services/document_upload.py's job-id shape (see
app/services/instance_proxy.py's own tests for why the id needs this
exact prefix — routing a later poll deterministically back to whichever
instance's in-memory _upload_jobs dict the job actually lives in) and
get_job_for_user's ownership check."""

import pytest

from app.services import document_upload


@pytest.mark.asyncio
async def test_start_upload_job_id_is_prefixed_with_this_instance_index(db, user, tmp_path, monkeypatch):
    monkeypatch.setattr(document_upload, "INSTANCE_INDEX", 3)

    status = await document_upload.start_upload_job(
        db,
        background_tasks=None,
        target_dir=tmp_path,
        owner_id=user.id,
        accepted=[],
        rejects=[],
        user=user,
    )

    assert status.job_id.startswith("3-")
    assert document_upload.get_job(status.job_id) is not None


@pytest.mark.asyncio
async def test_get_job_for_user_returns_none_for_someone_elses_job(
    db, user, admin_user, channel_manager_user, tmp_path
):
    """Security regression test: get_job (a bare dict lookup with no
    owner filter at all) used to be the only thing app/routers/documents.py
    called, so any authenticated user who learned another user's job_id
    could read their private upload filenames/errors. get_job_for_user
    is the fix — this asserts the ownership check actually works, not
    just that the function exists."""
    status = await document_upload.start_upload_job(
        db,
        background_tasks=None,
        target_dir=tmp_path,
        owner_id=user.id,
        accepted=[],
        rejects=[],
        user=user,
    )

    assert document_upload.get_job_for_user(status.job_id, user) is not None
    assert document_upload.get_job_for_user(status.job_id, admin_user) is not None  # admins can see any job
    assert document_upload.get_job_for_user(status.job_id, channel_manager_user) is None  # a different, plain user
    assert document_upload.get_job_for_user("does-not-exist", user) is None
