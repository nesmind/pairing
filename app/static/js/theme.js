/**
 * The app-wide UI theme picker's client-side logic (see app/static/css/themes.css for the actual color/font
 * tokens and app/services/theme_service.py for the persisted side). Loaded only from settings.html — every other
 * page is themed entirely by the server-rendered <html data-theme="..."> attribute (see app/main.py's
 * _profile_context) plus CSS, with zero script needed.
 *
 * First class-based JS in this codebase — mirrors the shape of app/services/auth_service.py's PageAuth: a small
 * constructor capturing current state, plain methods with one job each.
 */
class ThemeManager {
  constructor() {
    this.current = document.documentElement.dataset.theme;
    this.saved = this.current; // lets an unsaved live preview revert without a page reload
  }

  /** Instant, local-tab-only repaint — no network call. Safe to call on every swatch click for a live preview. */
  apply(themeId) {
    document.documentElement.dataset.theme = themeId;
    this.current = themeId;
  }

  /** Reverts an unsaved preview back to the last-saved theme (e.g. the user navigates away without saving). */
  revert() {
    this.apply(this.saved);
  }

  /** Persists `themeId` via PUT /api/settings/theme, then applies it and updates what revert() would fall back
   * to. Also mirrors it to localStorage so app.js's `storage` listener can reload this tab's other open tabs
   * onto the new theme — the same "other tabs pick it up" idea the code-theme picker already established,
   * upgraded here from "needs a manual refresh" to "reloads itself". */
  async save(themeId) {
    await api("/api/settings/theme", { method: "PUT", body: JSON.stringify({ theme: themeId }) });
    this.apply(themeId);
    this.saved = themeId;
    try {
      localStorage.setItem("pairing:uiTheme", themeId);
    } catch (_) {
      /* private-browsing/storage-disabled — cross-tab sync just doesn't happen, not fatal */
    }
  }
}

/**
 * Reads the active theme's CSS custom properties for the D3/SVG chart files, which write raw hex/rgb strings as
 * literal SVG attributes and can't use Tailwind classes at all. Cached per (theme, var) pair; no re-render
 * listener needed — every chart-rendering call site re-reads fresh data on its own page's load/poll cycle, so
 * the next natural render already picks up whatever theme is active by then.
 */
class ThemeColors {
  static _cache = new Map();

  /** Returns a CSS custom property's current value as a color string usable directly in a D3 `.attr("fill", ...)`
   * call (e.g. "56 189 248" from --chart-1, wrapped as "rgb(56 189 248)"). */
  static get(varName) {
    const theme = document.documentElement.dataset.theme;
    const cacheKey = `${theme}:${varName}`;
    if (ThemeColors._cache.has(cacheKey)) return ThemeColors._cache.get(cacheKey);
    const triplet = getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
    const value = `rgb(${triplet})`;
    ThemeColors._cache.set(cacheKey, value);
    return value;
  }

  /** --chart-1 through --chart-9 (see themes.css) — a themed, distinguishable color per chart series/category. */
  static chart(n) {
    return ThemeColors.get(`--chart-${n}`);
  }
}
