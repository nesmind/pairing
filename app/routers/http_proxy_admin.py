"""Settings > System > Outbound HTTP proxy: the admin-configured IP:port (optional username/password) for
internet-facing downloads: Ollama/ComfyUI installers (ollama_installer.py, comfyui_installer.py) and, in
local mode, Ollama's model pulls (ollama_process.py _build_env). Unrelated to proxy_admin.py, which
load-balances incoming requests across local instances ("Local pAIring server mode" in the UI).
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.schemas import HttpProxyConfig
from app.services import http_proxy_service
from app.services.auth_service import require_admin

router = APIRouter(prefix="/api/settings/http-proxy", tags=["http-proxy"])


@router.get("", response_model=HttpProxyConfig)
async def get_http_proxy(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    return await http_proxy_service.get_http_proxy_config_for_display(db)


@router.put("", response_model=HttpProxyConfig)
async def update_http_proxy(
    body: HttpProxyConfig,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """No broadcast needed: nothing caches it; every call site re-reads it from the DB. Returns a fresh masked
    read, never echoing the submitted password.
    """
    await http_proxy_service.set_http_proxy_config(db, body)
    return await http_proxy_service.get_http_proxy_config_for_display(db)
