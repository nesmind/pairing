"""Unit tests for app/services/avatar_service.py: profile picture
validation/storage/replacement/removal."""

import io

import pytest
from fastapi import HTTPException, UploadFile

from app.services import avatar_service


def _upload(filename: str, data: bytes) -> UploadFile:
    return UploadFile(io.BytesIO(data), filename=filename)


@pytest.fixture(autouse=True)
def _isolated_avatars_dir(tmp_path, monkeypatch):
    """save_avatar()/delete_avatar() write through app.config.AVATARS_DIR directly (same module-split
    import-binding gotcha as chat_attachment_service's identical fixture) — without this, every test below would
    touch the project's own real avatars/ folder."""
    monkeypatch.setattr(avatar_service, "AVATARS_DIR", tmp_path)


@pytest.mark.asyncio
async def test_save_avatar_writes_the_file_and_sets_avatar_path(db, user):
    updated = await avatar_service.save_avatar(db, user, _upload("me.png", b"fake-png-bytes"))

    # <user.id>-<random token>.png, not just <user.id>.png — see save_avatar's own docstring on why the random
    # component matters (it's what makes avatar_url's ?v= cache-buster below always genuinely change).
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

    assert not old_path.exists()  # the stale .jpg is gone, not left orphaned
    assert user.avatar_path.endswith(".png")
    assert (avatar_service.AVATARS_DIR / user.avatar_path).read_bytes() == b"new-png-bytes"


@pytest.mark.asyncio
async def test_reuploading_the_same_extension_still_changes_avatar_url(db, user):
    """The actual regression this covers: re-uploading a picture with the same extension used to write to the
    exact same filename — without a cache-busting suffix on avatar_url, the browser kept showing the stale
    cached image after a re-upload, confirmed live. avatar_url (and now avatar_path itself, see save_avatar's
    own docstring) must differ across two real, distinct uploads even when the extension doesn't. save_avatar
    mutates and returns the same User object both times, so each URL is captured as a plain string right after
    its own save — holding onto the object itself would just re-evaluate the property against whatever the
    *final* state ended up being."""
    updated = await avatar_service.save_avatar(db, user, _upload("me.png", b"first-bytes"))
    first_path, first_url = updated.avatar_path, updated.avatar_url
    updated = await avatar_service.save_avatar(db, user, _upload("me.png", b"second-bytes"))
    second_path, second_url = updated.avatar_path, updated.avatar_url

    assert first_path != second_path  # a genuinely different filename this time, not just a different URL
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
    await avatar_service.delete_avatar(user)  # must not raise
    assert user.avatar_path is None
