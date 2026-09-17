"""Request/response shapes for app/routers/conversations.py."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.schemas.common import GenerationParams


class ConversationCreate(BaseModel):
    model: str | None = None


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    model: str | None
    params: GenerationParams
    created_at: datetime
    updated_at: datetime


class ConversationUpdate(BaseModel):
    title: str | None = None
    model: str | None = None
    params: GenerationParams | None = None
