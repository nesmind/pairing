"""Unit tests for app/models/user.py's User.initials/avatar_url
properties — the small round-avatar fallback used wherever a picture
hasn't been uploaded (see app/services/avatar_service.py,
app/models/conversation.py's Message.sender_avatar_url/sender_initials,
and chat.js's renderAvatar)."""

from app.models import User


def test_initials_combines_first_and_last_name():
    user = User(username="asmith", first_name="Alice", last_name="Smith")
    assert user.initials == "AS"


def test_initials_uses_whichever_name_is_set():
    assert User(username="asmith", first_name="Alice").initials == "A"
    assert User(username="asmith", last_name="Smith").initials == "S"


def test_initials_falls_back_to_the_username_first_letter():
    user = User(username="asmith")
    assert user.initials == "A"


def test_avatar_url_is_none_until_a_picture_is_uploaded():
    user = User(id="u1", username="asmith")
    assert user.avatar_url is None


def test_avatar_url_points_at_the_account_avatar_endpoint_with_avatar_path_as_the_cache_buster():
    # avatar_path itself is unique per upload (a random token baked into the filename — see
    # avatar_service.save_avatar), so reusing it verbatim as the ?v= value needs no filesystem access at all.
    user = User(id="u1", username="asmith", avatar_path="u1-a1b2c3d4.png")
    assert user.avatar_url == "/api/account/u1/avatar?v=u1-a1b2c3d4.png"
