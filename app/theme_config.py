"""
Theme config — what's actually offered from Settings > Account >
Appearance, plus the fixed syntax-highlighting theme. Split out of
app/config.py (which was already at CLAUDE.md's 250-line cap) since
this is "what themes exist" data, not environment/deployment
configuration the way the rest of config.py is; moving it here also
reclaims the lines config.py needed.

- HLJS_THEME: the syntax-highlighting colors for code blocks in
  replies — a single fixed choice (not a per-user setting; that picker
  was removed once UI_THEMES below covered "customize how the app
  looks" more broadly, and a second independent picker for just code
  blocks stopped pulling its weight). Its id is also its filename under
  app/static/vendor/highlightjs/styles/<id>.min.css (see "Vendored
  frontend libraries" in README.md) — a real, unmodified highlight.js
  11.10.0 stylesheet, picked from its built-in theme set.
- UI_THEMES: the app-wide theme (backgrounds, text, accent colors,
  fonts — see app/static/css/themes.css and
  app/services/theme_service.py's get_ui_theme/set_ui_theme).
"""

HLJS_THEME = "nord"

# --- App-wide UI theme --------------------------------------------------
# Each entry's id is also the value app/templates/base.html writes to <html data-theme="..."> and the selector
# every block in app/static/css/themes.css is keyed by — see that file for the actual color/font values. `mode`
# is display-only (groups the Settings picker into "Dark"/"Light" rows), not stored or validated against anything.
# `swatch_bg`/`swatch_accent` are that same theme's --color-slate-800/--color-brand-500 values, hand-copied as
# plain hex so the Settings picker can render an accurate two-tone preview dot per theme in pure CSS/HTML, with
# no JS (and no temporarily applying a theme just to read its colors) needed just to draw the picker itself.
# "midnight" is this app's original hardcoded-dark look, kept first in this list for that historical reason —
# DEFAULT_UI_THEME below is a separate, independent choice (by value, not list position; see
# theme_service.get_ui_theme) of what a brand-new install/account that's never opened Appearance actually starts
# on, currently "nord" rather than this first entry.
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
