"""Tests for routers/account.py, called directly, not via TestClient/ASGI (see tests/test_instance_proxy_http.py
for why). Save/delete logic is covered by tests/test_avatar_service.py; this covers the router: profile
get/patch, avatar upload/delete, public avatar serving and the public profile lookup for chat.js's avatar modal.
"""

import io

import pytest
from fastapi import HTTPException, UploadFile

from app.routers import account
from app.schemas import AccountProfileUpdate
from app.services import avatar_service


def _upload(filename: str, data: bytes) -> UploadFile:
    return UploadFile(io.BytesIO(data), filename=filename)


@pytest.fixture(autouse=True)
def _isolated_avatars_dir(tmp_path, monkeypatch):
    # AVATARS_DIR is imported by name into both modules (independent bindings, the module-split gotcha), so patch
    # both for a save (avatar_service) and a serve (account's get_avatar) to agree on the file.
    monkeypatch.setattr(avatar_service, "AVATARS_DIR", tmp_path)
    monkeypatch.setattr(account, "AVATARS_DIR", tmp_path)


@pytest.mark.asyncio
async def test_get_profile_reads_the_current_users_own_fields(user):
    profile = await account.get_profile(user=user)
    assert profile.first_name is None
    assert profile.initials == user.username[0].upper()


@pytest.mark.asyncio
async def test_update_profile_sets_first_and_last_name(db, user):
    profile = await account.update_profile(
        AccountProfileUpdate(first_name="Alice", last_name="Smith"), db=db, user=user
    )
    assert profile.first_name == "Alice"
    assert profile.last_name == "Smith"
    assert profile.initials == "AS"


@pytest.mark.asyncio
async def test_update_profile_blank_clears_an_existing_name(db, user):
    await account.update_profile(AccountProfileUpdate(first_name="Alice", last_name="Smith"), db=db, user=user)
    profile = await account.update_profile(AccountProfileUpdate(first_name="", last_name=""), db=db, user=user)
    assert profile.first_name is None
    assert profile.last_name is None


@pytest.mark.asyncio
async def test_upload_avatar_saves_the_file_and_returns_the_updated_profile(db, user):
    profile = await account.upload_avatar(file=_upload("me.png", b"fake-bytes"), db=db, user=user)
    assert profile.avatar_url == f"/api/account/{user.id}/avatar?v={user.avatar_path}"


@pytest.mark.asyncio
async def test_remove_avatar_clears_the_picture(db, user):
    await account.upload_avatar(file=_upload("me.png", b"fake-bytes"), db=db, user=user)
    profile = await account.remove_avatar(db=db, user=user)
    assert profile.avatar_url is None
    assert profile.initials == user.username[0].upper()


@pytest.mark.asyncio
async def test_get_avatar_serves_another_users_uploaded_picture(db, user):
    await account.upload_avatar(file=_upload("me.png", b"fake-bytes"), db=db, user=user)

    response = await account.get_avatar(user_id=user.id, db=db, _user=None)

    assert str(response.path) == str(avatar_service.AVATARS_DIR / user.avatar_path)
    assert response.media_type == "image/png"


@pytest.mark.asyncio
async def test_get_avatar_404s_when_no_picture_is_set(db, user):
    with pytest.raises(HTTPException) as exc_info:
        await account.get_avatar(user_id=user.id, db=db, _user=None)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_get_avatar_404s_for_an_unknown_user_id(db):
    with pytest.raises(HTTPException) as exc_info:
        await account.get_avatar(user_id="does-not-exist", db=db, _user=None)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_get_public_profile_returns_another_users_basic_details(db, user):
    await account.update_profile(AccountProfileUpdate(first_name="Alice", last_name="Smith"), db=db, user=user)

    profile = await account.get_public_profile(user_id=user.id, db=db, _user=None)

    assert profile.username == user.username
    assert profile.first_name == "Alice"
    assert profile.last_name == "Smith"
    assert profile.initials == "AS"


@pytest.mark.asyncio
async def test_get_public_profile_404s_for_an_unknown_user_id(db):
    with pytest.raises(HTTPException) as exc_info:
        await account.get_public_profile(user_id="does-not-exist", db=db, _user=None)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_new_user_defaults_to_utc_timezone(user):
    profile = await account.get_profile(user=user)
    assert profile.timezone == "UTC"


@pytest.mark.asyncio
async def test_update_profile_sets_timezone_and_name_only_save_keeps_it(db, user):
    profile = await account.update_profile(AccountProfileUpdate(timezone="Asia/Jerusalem"), db=db, user=user)
    assert profile.timezone == "Asia/Jerusalem"
    profile = await account.update_profile(AccountProfileUpdate(first_name="Al"), db=db, user=user)
    assert profile.timezone == "Asia/Jerusalem"


def test_profile_update_rejects_unknown_timezone():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AccountProfileUpdate(timezone="Mars/Olympus_Mons")
