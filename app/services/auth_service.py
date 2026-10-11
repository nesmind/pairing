"""Authentication: password hashing, login throttling, the FastAPI dependencies requiring a user or admin,
and PageAuth (same check for HTML routes, which redirect instead of raising).

Sessions are Starlette's SessionMiddleware (app/main.py): a cookie holding `{"user_id":...}` signed
with SECRET_KEY, with no server-side table.

bcrypt calls stay blocking (CPU-bound, ~100ms, few concurrent users; CLAUDE.md's async carve-out).
Revisit for many concurrent logins.
"""

import time

import bcrypt
from fastapi import Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User

# Login throttling: in-memory client IP -> recent failure timestamps. Resets on restart and is not shared
# across workers; fine for single-server use.
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
    """The IP the rate limiter keys on. `forwarded_for` is trusted only when `direct_ip` is loopback, i.e. via
    instance_proxy.py's balancer (proxy_mode="local"), which sets the header. Otherwise an attacker could
    rotate X-Forwarded-For past the limit.
    """
    if direct_ip in ("127.0.0.1", "::1") and forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return direct_ip


def hash_password(password: str) -> str:
    """One-way bcrypt hash with its own random salt (equal passwords hash differently); compare via verify_password."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


# Placeholder hash so login still runs a bcrypt comparison for unknown usernames; otherwise the timing
# gap reveals valid usernames. See verify_password_or_dummy.
_DUMMY_PASSWORD_HASH = "$2b$12$HfzYiavSCrc.jixfSvznC.L4UWgTyyN.Sr198JT4ZQj35iCUotqlm"


def verify_password_or_dummy(password: str, password_hash: str | None) -> bool:
    """Like verify_password, but runs bcrypt even when `password_hash` is None (unknown user), so timing is
    the same. Returns False for None.
    """
    real_hash = password_hash if password_hash is not None else _DUMMY_PASSWORD_HASH
    ok = verify_password(password, real_hash)
    return ok if password_hash is not None else False


async def get_current_user(request: Request, db: AsyncSession = Depends(get_db)) -> User:
    """Dependency: the logged-in user from the session cookie. 401 if no session, deleted user or disabled
    account. Checking `status` here makes disabling take effect on the next request.
    """
    user_id = request.session.get("user_id")
    if user_id:
        user = await db.get(User, user_id)
        if user is not None and user.status == "active":
            return user
        request.session.clear()
    raise HTTPException(status_code=401, detail="Not authenticated")


def require_admin(user: User = Depends(get_current_user)) -> User:
    """Like get_current_user, but 403 for non-admins."""
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


class PageAuth:
    """Page-route counterpart of get_current_user/require_admin: same cookie check (stale sessions are
    cleared), but returns a RedirectResponse instead of raising.

    One instance per request: `await.resolve`, then `.require_login` or `.require_admin`, each
    returning the User or a RedirectResponse. Fails closed: `.user` stays None until `.resolve` finds a
    valid session, so forgetting it redirects to login instead of bypassing.

        auth = await PageAuth(request, db).resolve
        if isinstance(user:= auth.require_admin, RedirectResponse):
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
        """For pages that differ for admins without gating the whole page; use require_admin to block non-admins
        entirely."""
        return self.user is not None and self.user.role == "admin"

    def require_login(self) -> "User | RedirectResponse":
        return self.user if self.user is not None else RedirectResponse("/login")

    def require_admin(self) -> "User | RedirectResponse":
        if self.user is None:
            return RedirectResponse("/login")
        return self.user if self.is_admin else RedirectResponse("/")


async def seed_default_users(db: AsyncSession) -> User:
    """Creates the starter admin if missing (idempotent; safe every startup) and returns it, so app/main.py can
    backfill pre-multi-user conversations and db_config_service can seed a fresh database.

    Its password equals its username: fine for a first login on a private network, but change it before
    exposing the app (it is hashed, but "admin" is easy to guess).
    """
    existing = (await db.execute(select(User).where(User.username == "admin"))).scalar_one_or_none()
    if existing is None:
        db.add(
            User(
                username="admin",
                password_hash=hash_password("admin"),
                role="admin",
                first_name="Admin",
                last_name="Demo",
            )
        )
        await db.commit()
    return (await db.execute(select(User).where(User.username == "admin"))).scalar_one()
