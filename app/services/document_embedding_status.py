"""How the knowledge base relates to the current embedding model, for Settings > System > RAG."""

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Document
from app.schemas import EmbeddingStatus
from app.services import model_catalog_service


class DocumentEmbeddingStatus:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def get(self) -> EmbeddingStatus:
        current = await model_catalog_service.get_default_embedding_model(self._db)
        rows = (
            await self._db.execute(select(Document.embedding_model, func.count()).group_by(Document.embedding_model))
        ).all()
        counts = {model: count for model, count in rows}
        unknown = counts.pop(None, 0)
        matching = counts.pop(current, 0) if current else 0
        return EmbeddingStatus(
            current_model=current,
            total=unknown + matching + sum(counts.values()),
            current=matching,
            outdated=sum(counts.values()),
            unknown=unknown,
        )
