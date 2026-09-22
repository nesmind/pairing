"""Local server resource usage — see app/services/system_metrics_poller.py
for how these rows get written (a plain periodic psutil/nvidia-smi
sample, not an OpenTelemetry span like app/models/telemetry.py's
tables) and app/services/system_metrics_service.py for how the Stats
page's "System" view reads them back.
"""

from sqlalchemy import BigInteger, Column, DateTime, Float, Integer, String

from app.database import Base


class SystemMetricSnapshot(Base):
    """One periodic sample of this machine's CPU/RAM/disk usage — a
    pure append-only log, same reasoning as app.models.telemetry's two
    tables for using a plain autoincrementing id instead of this
    codebase's usual new_id() convention (nothing ever looks a row up
    by its own id from outside)."""

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
    """One periodic sample of one GPU's usage (see
    app.services.system_metrics_poller._sample_gpus — best-effort via
    `nvidia-smi`, an empty result on any machine without an NVIDIA GPU/
    driver, not an error). One row per (gpu_index, poll) — a machine
    with several GPUs gets several rows sharing the same polled_at,
    same append-only-log reasoning as SystemMetricSnapshot above."""

    __tablename__ = "gpu_metric_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    gpu_index = Column(Integer, nullable=False)
    name = Column(String(255), nullable=False)
    utilization_percent = Column(Float, nullable=False)
    mem_used_bytes = Column(BigInteger, nullable=False)
    mem_total_bytes = Column(BigInteger, nullable=False)
    temperature_c = Column(Float, nullable=True)
    polled_at = Column(DateTime, nullable=False)
