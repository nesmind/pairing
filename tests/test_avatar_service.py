"""Tests for services/avatar_service.py: profile picture validation, storage, replacement and removal."""

import io

import pytest
from fastapi import HTTPException, UploadFile

from app.services import avatar_service


def _upload(filename: str, data: bytes) -> UploadFile:
    return UploadFile(io.BytesIO(data), filename=filename)


@pytest.fixture(autouse=True)
def _isolated_avatars_dir(tmp_path, monkeypatch):
    """save_avatar/delete_avatar write through app.config.AVATARS_DIR (the same import-binding gotcha as
    chat_attachment_service's fixture); otherwise tests would touch the project's real avatars/ folder.
    """
    monkeypatch.setattr(avatar_service, "AVATARS_DIR", tmp_path)


@pytest.mark.asyncio
async def test_save_avatar_writes_the_file_and_sets_avatar_path(db, user):
    updated = await avatar_service.save_avatar(db, user, _upload("me.png", b"fake-png-bytes"))

    # <user.id>-<random token>.png: the random part makes avatar_url's ?v= cache-buster always change (see save_avatar).
    assert updated.avatar_path.startswith(f"{user.id}-")
    assert updated.avatar_path.endswith(".png")
    assert (avatar_service.AVATARS_DIR / updated.avatar_path).read_bytes() == b"fake-png-bytes"
    assert updated.avatar_url == f"/api/account/{user.id}/avatar?v={updated.avatar_path}"


@pytest.mark.asyncio
async def test_save_avatar_rejects_an_unsupported_extension(db, user):
    with pytest.raises(HTTPException) as exc_info:
        await avatar_service.save_avatar(db, user, _upload("me.exe", b"nope"))
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_save_avatar_rejects_a_file_over_the_size_limit(db, user, monkeypatch):
    monkeypatch.setattr(avatar_service, "MAX_AVATAR_MB", 1)
    with pytest.raises(HTTPException) as exc_info:
        await avatar_service.save_avatar(db, user, _upload("me.png", b"x" * (2 * 1024 * 1024)))
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_save_avatar_replaces_a_previously_uploaded_picture(db, user):
    await avatar_service.save_avatar(db, user, _upload("me.jpg", b"old-jpg-bytes"))
    old_path = avatar_service.AVATARS_DIR / user.avatar_path

    await avatar_service.save_avatar(db, user, _upload("me.png", b"new-png-bytes"))

    assert not old_path.exists()
    assert user.avatar_path.endswith(".png")
    assert (avatar_service.AVATARS_DIR / user.avatar_path).read_bytes() == b"new-png-bytes"


@pytest.mark.asyncio
async def test_reuploading_the_same_extension_still_changes_avatar_url(db, user):
    """Regression: re-uploading with the same extension reused the filename, and without a cache-buster the browser
    showed the stale image. avatar_url (and avatar_path) must differ between two uploads even with the same
    extension. save_avatar returns the same User object both times, so each URL is captured as a string right
    after its save (the object would re-evaluate against the final state).
    """
    updated = await avatar_service.save_avatar(db, user, _upload("me.png", b"first-bytes"))
    first_path, first_url = updated.avatar_path, updated.avatar_url
    updated = await avatar_service.save_avatar(db, user, _upload("me.png", b"second-bytes"))
    second_path, second_url = updated.avatar_path, updated.avatar_url

    assert first_path != second_path
    assert first_url != second_url


@pytest.mark.asyncio
async def test_delete_avatar_removes_the_file_and_clears_avatar_path(db, user):
    await avatar_service.save_avatar(db, user, _upload("me.png", b"fake-png-bytes"))
    saved_path = avatar_service.AVATARS_DIR / user.avatar_path

    await avatar_service.delete_avatar(user)

    assert not saved_path.exists()
    assert user.avatar_path is None


@pytest.mark.asyncio
async def test_delete_avatar_is_a_no_op_when_nothing_was_uploaded(db, user):
    await avatar_service.delete_avatar(user)
    assert user.avatar_path is None
