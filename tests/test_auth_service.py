"""Unit tests for app/services/auth_service.py: password hashing, the
timing-safe unknown-username path, the in-memory login rate limiter,
seed_default_users (the one DB-touching function here, used both by a
real app startup — see app/main.py — and by
app.services.db_config_service when building a fresh database from
Settings > System > Database), and PageAuth (the page-route login/
admin check app/main.py's HTML routes use)."""

import pytest
from fastapi.responses import RedirectResponse
from sqlalchemy import select

from app.models import User
from app.services import auth_service
from app.services.auth_service import PageAuth


def test_hash_and_verify_round_trip():
    hashed = auth_service.hash_password("correct horse battery staple")
    assert auth_service.verify_password("correct horse battery staple", hashed)
    assert not auth_service.verify_password("wrong password", hashed)


def test_hash_is_salted_differently_each_time():
    a = auth_service.hash_password("same-password")
    b = auth_service.hash_password("same-password")
    assert a != b


def test_verify_password_or_dummy_with_real_hash():
    hashed = auth_service.hash_password("hunter2")
    assert auth_service.verify_password_or_dummy("hunter2", hashed) is True
    assert auth_service.verify_password_or_dummy("wrong", hashed) is False


def test_verify_password_or_dummy_with_no_user_always_false():
    """The whole point of this function: an unknown username still runs
    a real bcrypt comparison (against the fixed dummy hash) but always
    reports False, regardless of what password was typed."""
    assert auth_service.verify_password_or_dummy("anything", None) is False
    assert auth_service.verify_password_or_dummy("", None) is False


def test_login_rate_limiter_trips_after_limit_and_clears():
    ip = "203.0.113.5"
    auth_service.clear_failed_logins(ip)  # isolate from any prior test's state

    for _ in range(auth_service._LOGIN_ATTEMPT_LIMIT - 1):
        assert not auth_service.is_login_rate_limited(ip)
        auth_service.record_failed_login(ip)

    # One more failure reaches the limit.
    auth_service.record_failed_login(ip)
    assert auth_service.is_login_rate_limited(ip)

    auth_service.clear_failed_logins(ip)
    assert not auth_service.is_login_rate_limited(ip)


def test_login_rate_limiter_is_per_ip():
    ip_a, ip_b = "203.0.113.10", "203.0.113.20"
    auth_service.clear_failed_logins(ip_a)
    auth_service.clear_failed_logins(ip_b)

    for _ in range(auth_service._LOGIN_ATTEMPT_LIMIT):
        auth_service.record_failed_login(ip_a)

    assert auth_service.is_login_rate_limited(ip_a)
    assert not auth_service.is_login_rate_limited(ip_b)


def test_resolve_client_ip_prefers_forwarded_for_only_from_loopback():
    # Came through the built-in proxy (app/services/instance_proxy.py) —
    # trust the header it set.
    assert auth_service.resolve_client_ip("127.0.0.1", "203.0.113.9") == "203.0.113.9"
    assert auth_service.resolve_client_ip("::1", "203.0.113.9") == "203.0.113.9"


def test_resolve_client_ip_ignores_forwarded_for_from_a_direct_remote_caller():
    # A remote attacker hitting /login directly can't spoof their way
    # past the rate limit just by setting their own header.
    assert auth_service.resolve_client_ip("203.0.113.9", "10.0.0.1") == "203.0.113.9"


def test_resolve_client_ip_falls_back_when_no_forwarded_for_header():
    assert auth_service.resolve_client_ip("127.0.0.1", None) == "127.0.0.1"


def test_resolve_client_ip_takes_the_first_hop_of_a_multi_value_header():
    assert auth_service.resolve_client_ip("127.0.0.1", "203.0.113.9, 10.0.0.1") == "203.0.113.9"


@pytest.mark.asyncio
async def test_seed_default_users_creates_exactly_one_admin_account(db):
    """Regression test: this used to also seed a second "ran"/"ran"
    account — a leftover personal/dev convenience, not a real default —
    onto every fresh install. Only the one admin account should exist
    now, with no name set (see User.first_name/last_name)."""
    admin = await auth_service.seed_default_users(db)

    all_users = (await db.execute(select(User))).scalars().all()
    assert [u.username for u in all_users] == ["admin"]

    assert admin.username == "admin"
    assert admin.role == "admin"
    assert auth_service.verify_password("admin", admin.password_hash)
    assert admin.first_name is None
    assert admin.last_name is None


@pytest.mark.asyncio
async def test_seed_default_users_is_idempotent(db):
    first = await auth_service.seed_default_users(db)
    second = await auth_service.seed_default_users(db)

    assert first.id == second.id
    all_users = (await db.execute(select(User))).scalars().all()
    assert len(all_users) == 1


class _FakeSession(dict):
    """Standing in for Starlette's real request.session — PageAuth only
    ever calls .get() and .clear() on it, both of which a plain dict
    already supports."""


class _FakeRequest:
    def __init__(self, user_id: str | None):
        self.session = _FakeSession({"user_id": user_id} if user_id else {})


@pytest.mark.asyncio
async def test_page_auth_with_no_session_redirects_both_checks_to_login(db):
    auth = await PageAuth(_FakeRequest(None), db).resolve()

    assert auth.user is None
    assert auth.is_admin is False
    login_result = auth.require_login()
    assert isinstance(login_result, RedirectResponse) and login_result.headers["location"] == "/login"
    admin_result = auth.require_admin()
    assert isinstance(admin_result, RedirectResponse) and admin_result.headers["location"] == "/login"


@pytest.mark.asyncio
async def test_page_auth_require_login_returns_the_real_user(db, user):
    auth = await PageAuth(_FakeRequest(user.id), db).resolve()

    assert auth.require_login() is user


@pytest.mark.asyncio
async def test_page_auth_require_admin_redirects_a_plain_user_to_chat_not_login(db, user):
    """Distinct from the not-logged-in-at-all case above: a logged-in non-admin is bounced to "/" (chat), never
    back to "/login" — they don't need to re-authenticate, just don't have the role for this page."""
    auth = await PageAuth(_FakeRequest(user.id), db).resolve()

    assert auth.is_admin is False
    result = auth.require_admin()
    assert isinstance(result, RedirectResponse) and result.headers["location"] == "/"


@pytest.mark.asyncio
async def test_page_auth_require_admin_returns_the_real_admin(db, admin_user):
    auth = await PageAuth(_FakeRequest(admin_user.id), db).resolve()

    assert auth.is_admin is True
    assert auth.require_admin() is admin_user


@pytest.mark.asyncio
async def test_page_auth_clears_a_session_pointing_at_a_deleted_user(db):
    request = _FakeRequest("some-id-that-was-never-created")

    auth = await PageAuth(request, db).resolve()

    assert auth.user is None
    assert "user_id" not in request.session


@pytest.mark.asyncio
async def test_page_auth_clears_a_session_for_a_disabled_user(db, user):
    user.status = "disabled"
    await db.commit()
    request = _FakeRequest(user.id)

    auth = await PageAuth(request, db).resolve()

    assert auth.user is None
    assert "user_id" not in request.session
    assert isinstance(auth.require_login(), RedirectResponse)


@pytest.mark.asyncio
async def test_page_auth_fails_closed_without_resolve(db, admin_user):
    """A route that forgot to `await .resolve()` must get a safe redirect, never an accidental bypass — there's
    no path where require_login()/require_admin() hand back a real User without a session actually being
    checked."""
    auth = PageAuth(_FakeRequest(admin_user.id), db)

    assert isinstance(auth.require_login(), RedirectResponse)
    assert isinstance(auth.require_admin(), RedirectResponse)
