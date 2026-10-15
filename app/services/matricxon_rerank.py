"""Reranking through Matricxon's POST /api/rerank (Knowledge base retrieval, document_retrieval.py)."""

import httpx

from app.services import matricxon_http, matricxon_pool
from app.services.matricxon_client import MatricxonError

_TIMEOUT = httpx.Timeout(120.0, connect=5.0)


class MatricxonRerankClient:
    async def rerank(self, query: str, documents: list[str], model: str) -> list[float]:
        """Scores in the same order as `documents` (Matricxon returns them best first)."""
        host = matricxon_pool.pick_host()
        async with matricxon_pool.track_request(host):
            async with matricxon_http.client(timeout=_TIMEOUT) as client:
                try:
                    resp = await client.post(
                        f"{host}/api/rerank", json={"model": model, "query": query, "documents": documents}
                    )
                    resp.raise_for_status()
                except httpx.HTTPError as exc:
                    if not isinstance(exc, httpx.HTTPStatusError):
                        matricxon_pool.mark_failure(host)
                    raise MatricxonError(f"Rerank request failed: {exc}") from exc
        matricxon_pool.mark_recovered(host)
        scores = [0.0] * len(documents)
        for result in resp.json()["results"]:
            scores[result["index"]] = float(result["relevance_score"])
        return scores
