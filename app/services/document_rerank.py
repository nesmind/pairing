"""Reorders similarity-search candidates with a reranker, when one is installed and switched on
(DefaultModelSettings.rerank). A reranker reads the question and each chunk together, so it judges
relevance better than vector similarity; failures keep the similarity order."""

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk
from app.services.default_model_settings import DefaultModelSettings
from app.services.engines.registry import registry
from app.services.inference_client import InferenceError, rerank

logger = logging.getLogger("llama_chat")


class ChunkReranker:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def order(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        """The best `top_k` of `chunks` (given best first by similarity)."""
        if len(chunks) < 2 or not registry.active().capabilities.rerank:
            return chunks[:top_k]
        model = await DefaultModelSettings(self._db).rerank()
        if model is None:
            return chunks[:top_k]
        try:
            scores = await rerank(query, [chunk.text for chunk in chunks], model)
        except InferenceError as exc:
            logger.warning("Reranking failed, keeping similarity order: %s", exc)
            return chunks[:top_k]
        ranked = sorted(zip(scores, range(len(chunks)), strict=True), reverse=True)
        return [chunks[index] for _score, index in ranked[:top_k]]
