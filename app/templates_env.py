"""The single Jinja2Templates instance every page route renders through (app/main.py page routes and
routers/auth.py's login page).

There used to be two instances, so a global registered on one (e.g. the default UI theme) silently didn't
exist on the other: Jinja rendered it as "", visible only when a CSS path broke. One instance, one place to register.
"""

from fastapi.templating import Jinja2Templates

from app.config import BASE_DIR
from app.theme_config import DEFAULT_UI_THEME, HLJS_THEME

templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))

# base.html's hljs theme <link>: a Jinja global so it can't drift from app.theme_config.HLJS_THEME.
templates.env.globals["hljs_theme"] = HLJS_THEME

# base.html's <html data-theme="..."> falls back to this on pages passing no `theme` (login.html).
templates.env.globals["default_ui_theme"] = DEFAULT_UI_THEME
