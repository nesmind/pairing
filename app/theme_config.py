"""Theme config: what Settings > Account > Appearance offers, plus the fixed syntax-highlighting theme. Split
from app/config.py (at the 250-line cap): "what themes exist" data, not deployment configuration.

- HLJS_THEME: code-block syntax colors, one fixed choice (the per-user picker was removed when UI_THEMES
  covered customization). Its id is also its filename under app/static/vendor/highlightjs/styles/<id>.min.css
  (README.md "Vendored frontend libraries"), an unmodified highlight.js 11.10.0 stylesheet.
- UI_THEMES: the app-wide theme (backgrounds, text, accents, fonts; static/css/themes.css,
  theme_service.get_ui_theme/set_ui_theme).
"""

HLJS_THEME = "nord"

# App-wide UI theme
# Each id is also the value base.html writes to <html data-theme="..."> and the selector for each block in
# static/css/themes.css (which holds the colors and fonts). `mode` is display-only (groups the picker into
# "Dark"/"Light"). `swatch_bg`/`swatch_accent` are the theme's --color-slate-800/--color-brand-500, hand-copied
# as hex so the picker draws a two-tone dot in pure CSS/HTML, with no JS or temporary theme application.
# "midnight" is the original dark look, kept first for history; DEFAULT_UI_THEME is an independent choice (by
# value, see theme_service.get_ui_theme) for new installs/accounts, currently "nord".
UI_THEMES = [
    {"id": "midnight", "label": "Midnight", "mode": "dark", "swatch_bg": "#1e293b", "swatch_accent": "#0a68ff"},
    {"id": "nord", "label": "Nord", "mode": "dark", "swatch_bg": "#292e38", "swatch_accent": "#499fb6"},
    {"id": "monokai", "label": "Monokai", "mode": "dark", "swatch_bg": "#33342d", "swatch_accent": "#f9065f"},
    {"id": "forest-dusk", "label": "Forest Dusk", "mode": "dark", "swatch_bg": "#273a34", "swatch_accent": "#8fcc33"},
    {"id": "ember", "label": "Ember", "mode": "dark", "swatch_bg": "#3d2c24", "swatch_accent": "#f24a0d"},
    {"id": "sunshine", "label": "Sunshine", "mode": "light", "swatch_bg": "#f0e7d6", "swatch_accent": "#f98806"},
    {"id": "bubblegum", "label": "Bubblegum", "mode": "light", "swatch_bg": "#edd9e6", "swatch_accent": "#e61980"},
    {"id": "sky", "label": "Sky", "mode": "light", "swatch_bg": "#d8e5ee", "swatch_accent": "#1392ec"},
    {"id": "meadow", "label": "Meadow", "mode": "light", "swatch_bg": "#dbebe0", "swatch_accent": "#39c66d"},
    {"id": "sand", "label": "Sand", "mode": "light", "swatch_bg": "#ebe3db", "swatch_accent": "#c66d39"},
]
DEFAULT_UI_THEME = "nord"
