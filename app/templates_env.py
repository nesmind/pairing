"""
The single Jinja2Templates instance every page route renders through —
app/main.py's page routes and app/routers/auth.py's login page both
import `templates` from here rather than each constructing their own.

This used to be two separate Jinja2Templates instances (one per file),
which meant a global registered on one (e.g. the default UI theme
fallback below) silently never existed on the other's environment —
not an error, just Jinja quietly rendering the undefined global as an
empty string, which only became visible once something (a CSS path)
actually broke when it came out empty. One shared instance means one
place to register a global and nowhere for the two to drift apart
again.
"""

from fastapi.templating import Jinja2Templates

from app.config import BASE_DIR
from app.theme_config import DEFAULT_UI_THEME, HLJS_THEME

templates = Jinja2Templates(directory=str(BASE_DIR / "app" / "templates"))

# base.html's hljs theme <link> — a Jinja global rather than a hardcoded string in the template, so it can never
# drift out of sync with app.theme_config.HLJS_THEME.
templates.env.globals["hljs_theme"] = HLJS_THEME

# base.html's <html data-theme="..."> falls back to this on pages that
# don't pass their own `theme` (login.html).
templates.env.globals["default_ui_theme"] = DEFAULT_UI_THEME
