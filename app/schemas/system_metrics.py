"""Response shapes for GET /api/system-metrics/summary — the Stats
page's "System" view (see app/services/system_metrics_service.py for
the query and app/static/js/system_metrics.js for how these render)."""

from datetime import datetime

from pydantic import BaseModel


class SystemMetricPoint(BaseModel):
    polled_at: datetime
    cpu_percent: float
    mem_percent: float


class GpuStat(BaseModel):
    """One GPU's current usage — see app.services.system_metrics_poller's
    best-effort `nvidia-smi` sampling. gpus is simply empty on any
    machine without an NVIDIA GPU/driver, not an error."""

    gpu_index: int
    name: str
    utilization_percent: float
    mem_used_bytes: int
    mem_total_bytes: int
    temperature_c: float | None
    polled_at: datetime


class GpuHistoryPoint(BaseModel):
    """Flat (not grouped by GPU) so a multi-GPU history stays one simple list — the frontend groups by
    gpu_index itself when drawing one line per GPU (see app/static/js/system_charts.js)."""

    polled_at: datetime
    gpu_index: int
    utilization_percent: float


class SystemMetricsSummary(BaseModel):
    cpu_percent: float
    cpu_count: int
    mem_used_bytes: int
    mem_total_bytes: int
    disk_used_bytes: int
    disk_total_bytes: int
    load_avg_1m: float | None
    polled_at: datetime | None
    history: list[SystemMetricPoint]
    gpus: list[GpuStat]
    gpu_history: list[GpuHistoryPoint]
