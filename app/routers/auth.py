"""
Login/logout. Deliberately plain HTML-form endpoints (not JSON API)
since a login page needs to work before any JavaScript-driven API calls
make sense — the browser posts the form directly and gets a redirect
back, the same way login has worked since before SPAs existed.
"""

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.services.auth_service import (
    clear_failed_logins,
    is_login_rate_limited,
    record_failed_login,
    resolve_client_ip,
    verify_password_or_dummy,
)
from app.templates_env import templates

router = APIRouter(tags=["auth"])


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    if request.session.get("user_id"):
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse(request, "login.html")


@router.post("/login", response_class=HTMLResponse)
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    direct_ip = request.client.host if request.client else "unknown"
    ip = resolve_client_ip(direct_ip, request.headers.get("x-forwarded-for"))
    if is_login_rate_limited(ip):
        return templates.TemplateResponse(
            request,
            "login.html",
            {"error": "Too many failed login attempts. Try again in a few minutes."},
            status_code=429,
        )
    user = (await db.execute(select(User).where(User.username == username))).scalar_one_or_none()
    password_ok = verify_password_or_dummy(password, user.password_hash if user else None)
    if user is None or not password_ok:
        record_failed_login(ip)
        return templates.TemplateResponse(
            request,
            "login.html",
            {"error": "Incorrect username or password."},
            status_code=401,
        )
    if user.status != "active":
        return templates.TemplateResponse(
            request,
            "login.html",
            {"error": "This account has been disabled. Contact an admin."},
            status_code=403,
        )
    clear_failed_logins(ip)
    request.session["user_id"] = user.id
    return RedirectResponse("/", status_code=303)


@router.post("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)
