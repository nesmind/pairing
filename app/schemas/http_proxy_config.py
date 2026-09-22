"""
Schemas for Settings > System > Outbound HTTP proxy — see
app/services/http_proxy_service.py for the get/set logic and
app/routers/http_proxy_admin.py for the admin endpoints. Deliberately
named distinctly from app.schemas.instances.ProxyMode: that's an
unrelated, pre-existing concept (whether this app's own *incoming* HTTP
requests get load-balanced across local sibling instances — labeled
"Local pAIring server mode" in the UI specifically to avoid this exact
collision). This one is about outbound: whether this app's own genuinely
internet-facing downloads (Ollama/ComfyUI installers, and Ollama's own
model pulls when it runs in local mode) go through an admin-configured
HTTP proxy.
"""

from urllib.parse import quote

from pydantic import BaseModel, Field


class HttpProxyConfig(BaseModel):
    """Admin-configured from Settings > System. No scheme picker — always
    constructed as http://{host}:{port} (see proxy_url below), matching
    what was actually asked for: a plain IP:port. `username`/`password`
    are optional — some proxies require them, most on a private LAN
    don't. `enabled` is stored separately from host/port (rather than
    "empty host means disabled") so a temporarily-disabled proxy doesn't
    lose its saved host/port.

    `password` is write-only in practice: app.services.http_proxy_service's
    get_http_proxy_config_for_display redacts it (via `has_password`
    instead) before this ever reaches a browser — same "never re-display
    a saved secret" precedent as app.schemas.db_config.MysqlParams/
    DbConfig. Only the *internal* get_http_proxy_config (used by the
    installer/process call sites that actually need a working proxy_url)
    ever sees the real value."""

    enabled: bool = False
    host: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    username: str | None = None
    password: str | None = None
    # Read-only/informational — set by get_http_proxy_config_for_display;
    # meaningless as input (a PUT never needs to send it).
    has_password: bool = False

    def proxy_url(self) -> str | None:
        """http://[user[:pass]@]host:port when usable, else None — every
        call site just does `if config.proxy_url(): ...` without
        re-deriving this same enabled-and-complete check itself.
        Username/password are percent-encoded (safe="") since either can
        contain characters (":", "@", "/") that would otherwise corrupt
        the URL."""
        if not (self.enabled and self.host and self.port):
            return None
        if self.username:
            auth = quote(self.username, safe="")
            if self.password:
                auth += f":{quote(self.password, safe='')}"
            return f"http://{auth}@{self.host}:{self.port}"
        return f"http://{self.host}:{self.port}"
