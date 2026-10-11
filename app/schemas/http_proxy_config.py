"""Schemas for Settings > System > Outbound HTTP proxy (services/http_proxy_service.py,
routers/http_proxy_admin.py). Outbound only: internet-facing downloads (Ollama/ComfyUI installers,
Ollama pulls in local mode); unrelated to instances.ProxyMode, which load-balances incoming requests.
"""

from urllib.parse import quote

from pydantic import BaseModel, Field


class HttpProxyConfig(BaseModel):
    """Admin-configured. Always http://{host}:{port} (no scheme picker). `username`/`password` are optional.
    `enabled` is separate from host/port so disabling keeps them.

    `password` is write-only in practice: get_http_proxy_config_for_display redacts it (`has_password`),
    as with db_config.MysqlParams/DbConfig. Only the internal get_http_proxy_config sees the real value.
    """

    enabled: bool = False
    host: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    username: str | None = None
    password: str | None = None
    # Informational, set by get_http_proxy_config_for_display; never sent on PUT.
    has_password: bool = False

    def proxy_url(self) -> str | None:
        """http://[user[:pass]@]host:port when enabled and complete, else None. Credentials are percent-encoded
        (safe="") since ":", "@" or "/" would corrupt the URL.
        """
        if not (self.enabled and self.host and self.port):
            return None
        if self.username:
            auth = quote(self.username, safe="")
            if self.password:
                auth += f":{quote(self.password, safe='')}"
            return f"http://{auth}@{self.host}:{self.port}"
        return f"http://{self.host}:{self.port}"
