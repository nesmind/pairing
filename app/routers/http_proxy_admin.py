"""
Settings > System > Outbound HTTP proxy admin endpoints: the single
admin-configured IP:port (with optional username/password) that this
app's own genuinely internet-facing downloads should route through —
Ollama's and ComfyUI's own GitHub-release/git-clone/pip installers (see
app/services/ollama_installer.py, app/services/comfyui_installer.py) and,
only when Ollama runs in local mode, Ollama's own outbound model-pull
requests (see app/services/ollama_process.py's _build_env). Distinct
from app/routers/proxy_admin.py, an unrelated pre-existing feature about
load-balancing this app's own incoming requests across local instances
(labeled "Local pAIring server mode" in the UI specifically to avoid
this naming collision).
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
    """No internal-refresh/broadcast needed the way Ollama/ComfyUI's own
    host-pool config gets — nothing caches this in-process; every call
    site (the installer endpoints, ollama_process.start/apply_local_config)
    re-fetches it fresh from the DB per action. Returns a fresh masked
    re-read (never the raw submitted body) so a real password an admin
    just typed is never echoed straight back."""
    await http_proxy_service.set_http_proxy_config(db, body)
    return await http_proxy_service.get_http_proxy_config_for_display(db)
