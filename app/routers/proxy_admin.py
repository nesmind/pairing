"""
Settings > System > Proxy mode admin endpoints: whether this app
load-balances across local instances itself ("local" — the default) or
leaves that to an external reverse proxy, if any ("proxy"). See
app/services/instance_pool.py + instance_proxy.py for the built-in
balancer this toggles. Changes here take effect immediately (the
primary's cached copy of the mode is updated in the same request), no
restart required.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.schemas import ProxyMode
from app.services import instance_pool, settings_service
from app.services.auth_service import require_admin

router = APIRouter(prefix="/api/settings/proxy-mode", tags=["proxy-mode"])


@router.get("", response_model=ProxyMode)
async def get_proxy_mode(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    return ProxyMode(mode=await settings_service.get_proxy_mode(db))


@router.put("", response_model=ProxyMode)
async def update_proxy_mode(
    body: ProxyMode,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    await settings_service.set_proxy_mode(db, body.mode)
    instance_pool.set_cached_proxy_mode(body.mode)
    return body
