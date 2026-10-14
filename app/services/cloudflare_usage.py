"""What the External servers tab shows about the Cloudflare account: token status and Workers AI usage
(Cloudflare's GraphQL analytics, dataset aiInferenceAdaptiveGroups). Usage needs the token to also have the
"Account Analytics: Read" permission; without it the tab says so instead."""

from datetime import UTC, datetime, timedelta

import httpx

from app.services import connector_config_cache

_GRAPHQL = "https://api.cloudflare.com/client/v4/graphql"
_VERIFY = "https://api.cloudflare.com/client/v4/user/tokens/verify"
_TIMEOUT = httpx.Timeout(15.0, connect=5.0)
_TOP_MODELS = 5
_NO_ANALYTICS = (
    'To see usage here, give the API token the "Account Analytics: Read" permission too '
    "(Cloudflare dashboard > My Profile > API Tokens > edit the token)."
)
_QUERY = """query($account: String!, $week: Time!, $today: Time!) { viewer { accounts(filter: {accountTag: $account}) {
  week: aiInferenceAdaptiveGroups(limit: 100, filter: {datetime_geq: $week}, orderBy: [sum_totalNeurons_DESC]) {
    count sum { totalNeurons totalInputTokens totalOutputTokens } dimensions { modelId } }
  today: aiInferenceAdaptiveGroups(limit: 1, filter: {datetime_geq: $today}) { count sum { totalNeurons } }
} } }"""


class CloudflareAccountReport:
    """Reads token status and usage for the saved Cloudflare config; never raises, so the tab always renders."""

    def __init__(self, config: dict[str, str] | None = None, now: datetime | None = None) -> None:
        self._config = config if config is not None else connector_config_cache.get_config("cloudflare")
        self._now = now or datetime.now(UTC)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._config.get('api_token', '').strip()}"}

    async def account_details(self) -> list[tuple[str, str]]:
        if not self._config.get("api_token"):
            return []
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(_VERIFY, headers=self._headers())
            status = (resp.json().get("result") or {}).get("status", "unknown")
        except (httpx.HTTPError, ValueError):
            status = "could not check"
        return [("Token status", status.capitalize()), ("Free allowance", "10,000 neurons a day")]

    def _variables(self) -> dict[str, str]:
        today = self._now.replace(hour=0, minute=0, second=0, microsecond=0)
        stamp = "%Y-%m-%dT%H:%M:%SZ"
        return {
            "account": self._config.get("account_id", "").strip(),
            "week": (self._now - timedelta(days=7)).strftime(stamp),
            "today": today.strftime(stamp),
        }

    @staticmethod
    def _shape(account: dict) -> dict:
        week = account.get("week") or []
        today = (account.get("today") or [{}])[0] if account.get("today") else {}
        total = lambda key: sum((row.get("sum") or {}).get(key) or 0 for row in week)  # noqa: E731
        return {
            "period": "Last 7 days",
            "units_label": "neurons",
            "requests": sum(row.get("count") or 0 for row in week),
            "units": round(total("totalNeurons"), 1),
            "input_tokens": int(total("totalInputTokens")),
            "output_tokens": int(total("totalOutputTokens")),
            "today_requests": today.get("count") or 0,
            "today_units": round((today.get("sum") or {}).get("totalNeurons") or 0, 1),
            "top_models": [
                {
                    "model": (row.get("dimensions") or {}).get("modelId", "?"),
                    "requests": row.get("count") or 0,
                    "units": round((row.get("sum") or {}).get("totalNeurons") or 0, 1),
                }
                for row in week[:_TOP_MODELS]
            ],
        }

    async def usage_summary(self) -> tuple[dict | None, str | None]:
        if not self._config.get("api_token") or not self._config.get("account_id"):
            return None, None
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.post(
                    _GRAPHQL, headers=self._headers(), json={"query": _QUERY, "variables": self._variables()}
                )
            body = resp.json()
        except (httpx.HTTPError, ValueError):
            return None, "Could not reach Cloudflare's analytics right now."
        errors = body.get("errors") or []
        if any((e.get("extensions") or {}).get("code") == "authz" for e in errors):
            return None, _NO_ANALYTICS
        accounts = ((body.get("data") or {}).get("viewer") or {}).get("accounts") or []
        if errors or not accounts:
            return None, "Cloudflare did not return usage for this account."
        return self._shape(accounts[0]), None
