"""Response shapes for GET /api/telemetry/ollama/summary — the
Telemetry page's Ollama dashboard (see app/services/telemetry_service.py
for the aggregation queries and app/static/js/telemetry.js for how
these render)."""

from datetime import datetime

from pydantic import BaseModel

from app.schemas.stats import CountByKey


class HostStats(BaseModel):
    host: str
    request_count: int
    error_count: int


class ModelLatencyStats(BaseModel):
    model: str
    avg_duration_ms: float
    p95_duration_ms: float
    request_count: int


class LatencyTimeBucket(BaseModel):
    """One hour-wide bucket — bucket_start is an ISO-ish "YYYY-MM-DDTHH:00:00" string, computed in Python
    (see telemetry_service's own docstring on why this isn't a SQL group_by)."""

    bucket_start: str
    avg_duration_ms: float
    request_count: int


class LoadedModelSnapshot(BaseModel):
    host: str
    model_name: str
    size_bytes: int | None
    size_vram_bytes: int | None
    expires_at: datetime | None
    polled_at: datetime


class OllamaTelemetrySummary(BaseModel):
    total_requests: int
    error_count: int
    by_model: list[CountByKey]
    by_host: list[HostStats]
    latency_by_model: list[ModelLatencyStats]
    latency_time_series: list[LatencyTimeBucket]
    loaded_models: list[LoadedModelSnapshot]
