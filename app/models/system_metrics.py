"""Local server resource usage, written by app/services/system_metrics_poller.py (periodic psutil/nvidia-smi
samples, not OpenTelemetry spans) and read by system_metrics_service.py for the Stats "System" view.
"""

from sqlalchemy import BigInteger, Column, DateTime, Float, Integer, String

from app.database import Base


class SystemMetricSnapshot(Base):
    """One periodic CPU/RAM/disk sample. Append-only, so a plain autoincrement id (as in app.models.telemetry)."""

    __tablename__ = "system_metric_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cpu_percent = Column(Float, nullable=False)
    mem_used_bytes = Column(BigInteger, nullable=False)
    mem_total_bytes = Column(BigInteger, nullable=False)
    disk_used_bytes = Column(BigInteger, nullable=False)
    disk_total_bytes = Column(BigInteger, nullable=False)
    load_avg_1m = Column(Float, nullable=True)
    polled_at = Column(DateTime, nullable=False)


class GpuMetricSnapshot(Base):
    """One periodic sample of one GPU (system_metrics_poller._sample_gpus; best-effort via `nvidia-smi`, empty
    without an NVIDIA GPU). One row per (gpu_index, poll), sharing polled_at.
    """

    __tablename__ = "gpu_metric_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    gpu_index = Column(Integer, nullable=False)
    name = Column(String(255), nullable=False)
    utilization_percent = Column(Float, nullable=False)
    mem_used_bytes = Column(BigInteger, nullable=False)
    mem_total_bytes = Column(BigInteger, nullable=False)
    temperature_c = Column(Float, nullable=True)
    polled_at = Column(DateTime, nullable=False)
