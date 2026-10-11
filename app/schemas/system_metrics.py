"""Shapes for GET /api/system-metrics/summary, the Stats "System" view (system_metrics_service.py,
system_metrics.js)."""

from datetime import datetime

from pydantic import BaseModel


class SystemMetricPoint(BaseModel):
    polled_at: datetime
    cpu_percent: float
    mem_percent: float


class GpuStat(BaseModel):
    """One GPU's current usage (best-effort `nvidia-smi` in system_metrics_poller); gpus is empty without an NVIDIA
    GPU."""

    gpu_index: int
    name: str
    utilization_percent: float
    mem_used_bytes: int
    mem_total_bytes: int
    temperature_c: float | None
    polled_at: datetime


class GpuHistoryPoint(BaseModel):
    """Flat so multi-GPU history stays one list; the frontend groups by gpu_index (system_charts.js)."""

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
