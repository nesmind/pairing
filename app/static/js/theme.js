/*
 * Client-side logic of the UI theme picker (tokens: app/static/css/themes.css; persistence:
 * app/services/theme_service.py). Loaded only from settings.html: other pages are themed by the server-rendered <html
 * data-theme="..."> attribute (app/main.py _profile_context) plus CSS, with no script.
 * 
 * The codebase's first class-based JS, shaped like app/services/auth_service.py's PageAuth: a constructor capturing
 * state, plain one-job methods.
 */
class ThemeManager {
  constructor() {
    this.current = document.documentElement.dataset.theme;
    this.saved = this.current;
  }

  /* Instant local repaint, no network call; safe on every swatch click for live preview. */
  apply(themeId) {
    document.documentElement.dataset.theme = themeId;
    this.current = themeId;
  }

  /* Reverts an unsaved preview to the last-saved theme (e.g. navigating away without saving). */
  revert() {
    this.apply(this.saved);
  }

  /* Persists `themeId` via PUT /api/settings/theme, applies it and updates what revert falls back to. Also mirrors it to localStorage so app.js's `storage` listener reloads other open tabs onto the new theme (as the code-theme picker, but self-refreshing). */
  async save(themeId) {
    await api("/api/settings/theme", { method: "PUT", body: JSON.stringify({ theme: themeId }) });
    this.apply(themeId);
    this.saved = themeId;
    try {
      localStorage.setItem("pairing:uiTheme", themeId);
    } catch (_) {
      /* storage disabled: cross-tab sync just doesn't happen */
    }
  }
}

/* Reads the active theme's CSS custom properties for the D3/SVG chart files, which write raw hex/rgb strings as SVG attributes and can't use Tailwind classes. Cached per (theme, var); no re-render listener is needed since each chart call site re-reads data on its page's load/poll cycle. */
class ThemeColors {
  static _cache = new Map();

  /* Returns a CSS custom property as a color string usable in D3 `.attr("fill",...)` (e.g. "56 189 248" from --chart-1 becomes "rgb(56 189 248)"). */
  static get(varName) {
    const theme = document.documentElement.dataset.theme;
    const cacheKey = `${theme}:${varName}`;
    if (ThemeColors._cache.has(cacheKey)) return ThemeColors._cache.get(cacheKey);
    const triplet = getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
    const value = `rgb(${triplet})`;
    ThemeColors._cache.set(cacheKey, value);
    return value;
  }

  /* --chart-1 through --chart-9 (themes.css): a themed, distinguishable color per chart series/category. */
  static chart(n) {
    return ThemeColors.get(`--chart-${n}`);
  }
}
