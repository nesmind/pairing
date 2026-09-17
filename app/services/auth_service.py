"""
Authentication: password hashing, login brute-force throttling, the
FastAPI dependencies API routers use to require a logged-in user (or
specifically an admin), and PageAuth — the same check for HTML page
routes, which need a redirect instead of a raised HTTPException.

Sessions are handled by Starlette's SessionMiddleware (registered in
app/main.py) — a signed cookie holding `{"user_id": ...}`. There's no
server-side session table: the cookie itself, signed with
app.config.SECRET_KEY, is the only state, which is enough for a
single-server self-hosted app and keeps this file simple.

bcrypt's hash/verify calls are left as plain blocking calls rather than
wrapped in asyncio.to_thread: they're CPU-bound (not I/O), take on the
order of 100ms on modest hardware, and this app targets a small number
of concurrent self-hosted users — the "unless dealing with strictly
blocking...I/O" carve-out in CLAUDE.md's async rule is exactly this kind
of tradeoff. Revisit if this ever needs to serve many concurrent logins.
"""

import time

import bcrypt
from fastapi import Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User

# Login brute-force throttling: an in-memory map of client IP -> timestamps
# of recent failed attempts. Resets on process restart and isn't shared
# across multiple worker processes, same tradeoff as the session cookie
# above — fine for this app's single-server deployment model, not a
# distributed rate limiter.
_LOGIN_ATTEMPT_WINDOW_SECONDS = 15 * 60
_LOGIN_ATTEMPT_LIMIT = 5
_failed_login_attempts: dict[str, list[float]] = {}


def is_login_rate_limited(ip: str) -> bool:
    now = time.monotonic()
    recent = [t for t in _failed_login_attempts.get(ip, []) if now - t < _LOGIN_ATTEMPT_WINDOW_SECONDS]
    _failed_login_attempts[ip] = recent
    return len(recent) >= _LOGIN_ATTEMPT_LIMIT


def record_failed_login(ip: str) -> None:
    _failed_login_attempts.setdefault(ip, []).append(time.monotonic())


def clear_failed_logins(ip: str) -> None:
    _failed_login_attempts.pop(ip, None)


def resolve_client_ip(direct_ip: str, forwarded_for: str | None) -> str:
    """The IP the login rate limiter above should actually key on.
    Only trusts `forwarded_for` when `direct_ip` is this machine's own
    loopback address — i.e. the request actually came through
    app/services/instance_proxy.py's built-in load balancer (Settings >
    System, proxy_mode="local"), which always connects from 127.0.0.1
    and sets this header itself. A remote attacker hitting /login
    directly could otherwise set their own X-Forwarded-For and freely
    rotate past the rate limit — trusting a client-supplied header for
    security-relevant IP attribution is only ever safe when it's known
    to come from a trusted hop, never from an arbitrary caller."""
    if direct_ip in ("127.0.0.1", "::1") and forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return direct_ip


def hash_password(password: str) -> str:
    """One-way hash for storing a password. bcrypt includes its own
    random salt, so the same password hashes differently every time —
    that's expected and correct, not a bug to "fix" by comparing hashes
    directly (always use verify_password instead)."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


# Not a real account's hash — a fixed placeholder so login can run a bcrypt
# comparison against *something* when the username doesn't exist. Without
# this, an unknown username short-circuits before bcrypt.checkpw ever runs
# while a real one takes bcrypt's tens-of-ms, and that timing gap lets an
# attacker enumerate valid usernames before even attempting to brute-force
# passwords. See verify_password_or_dummy below.
_DUMMY_PASSWORD_HASH = "$2b$12$HfzYiavSCrc.jixfSvznC.L4UWgTyyN.Sr198JT4ZQj35iCUotqlm"


def verify_password_or_dummy(password: str, password_hash: str | None) -> bool:
    """Like verify_password, but always runs bcrypt even when there's no
    real hash to check against (password_hash is None) — pass this the
    looked-up user's hash, or None when no such user exists, so a login
    attempt for an unknown username costs the same time as one for a real
    account. Always returns False when password_hash is None."""
    real_hash = password_hash if password_hash is not None else _DUMMY_PASSWORD_HASH
    ok = verify_password(password, real_hash)
    return ok if password_hash is not None else False


async def get_current_user(request: Request, db: AsyncSession = Depends(get_db)) -> User:
    """FastAPI dependency: the logged-in user, from the session cookie.
    Raises 401 if there's no session, it points at a since-deleted user,
    or the account has since been disabled — routers that need a
    logged-in user just add `Depends(get_current_user)` and get a real
    User back, no manual checking required.

    Checking `status` here (not just at login) means disabling a user
    takes effect immediately, on their very next request, rather than
    only blocking their *next* login while an already-open session of
    theirs keeps working."""
    user_id = request.session.get("user_id")
    if user_id:
        user = await db.get(User, user_id)
        if user is not None and user.status == "active":
            return user
        request.session.clear()
    raise HTTPException(status_code=401, detail="Not authenticated")


def require_admin(user: User = Depends(get_current_user)) -> User:
    """Like get_current_user, but 403s anyone who isn't an admin —
    for the Settings page's System tab and its endpoints."""
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


class PageAuth:
    """The page-route counterpart to get_current_user/require_admin above: same session-cookie check (including
    the same self-healing behavior — a stale session pointing at a deleted/disabled user gets cleared, not just
    rejected), but a raw 401/403 error page is a bad experience for someone who just clicked a sidebar link, so
    every check here returns a RedirectResponse for the route to hand straight back instead of raising.

    One instance per request — construct it, `await .resolve()`, then call `.require_login()` or
    `.require_admin()`, each returning either the real User or the RedirectResponse to return. Fails closed by
    design: `.user` starts (and stays) None until `.resolve()` actually finds a valid session, so a route that
    forgets to await `.resolve()` first gets an incorrect-but-safe "redirect to login" rather than an accidental
    bypass — there's no code path where a real User comes back without the session actually having been checked.

    Typical call site (see app/main.py's page routes, the only callers):

        auth = await PageAuth(request, db).resolve()
        if isinstance(user := auth.require_admin(), RedirectResponse):
            return user
    """

    def __init__(self, request: Request, db: AsyncSession) -> None:
        self._request = request
        self._db = db
        self.user: User | None = None

    async def resolve(self) -> "PageAuth":
        user_id = self._request.session.get("user_id")
        if user_id:
            user = await self._db.get(User, user_id)
            if user is not None and user.status == "active":
                self.user = user
                return self
            self._request.session.clear()
        return self

    @property
    def is_admin(self) -> bool:
        """For a page that renders differently for an admin without gating the whole page on it (e.g. chat.html's
        Stats sidebar link, settings.html's System tab) — use require_admin() instead when the *entire* page
        should be inaccessible to a non-admin."""
        return self.user is not None and self.user.role == "admin"

    def require_login(self) -> "User | RedirectResponse":
        return self.user if self.user is not None else RedirectResponse("/login")

    def require_admin(self) -> "User | RedirectResponse":
        if self.user is None:
            return RedirectResponse("/login")
        return self.user if self.is_admin else RedirectResponse("/")


async def seed_default_users(db: AsyncSession) -> User:
    """Creates the one starter admin account if it doesn't already
    exist — idempotent, so it's safe to call on every startup. Returns
    that admin User (existing or newly created), so app/main.py can use
    its id to backfill any conversation created before multi-user
    support existed, and app.services.db_config_service can seed a
    fresh database the same way a real first boot does.

    The account uses its username as its password. That's exactly as
    insecure as it sounds — fine for a first login on a private network,
    not fine to leave as-is on anything internet-reachable. The password
    is still stored properly hashed (never plaintext), but "properly
    hashed" doesn't help against someone simply typing "admin" — change
    it before exposing this app beyond your own machine.
    """
    existing = (await db.execute(select(User).where(User.username == "admin"))).scalar_one_or_none()
    if existing is None:
        db.add(User(username="admin", password_hash=hash_password("admin"), role="admin"))
        await db.commit()
    return (await db.execute(select(User).where(User.username == "admin"))).scalar_one()
